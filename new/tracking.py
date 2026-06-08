import cv2
import numpy as np
import csv
import os
from collections import defaultdict, deque, Counter
from ultralytics import YOLO
import torch

# ══════════════════════════════════════════════════════
#  CONFIG — change these
# ══════════════════════════════════════════════════════
INPUT_VIDEO  = "/home/maaz/eysip/new/video1.mp4"
OUTPUT_VIDEO = "/home/maaz/eysip/new/output_trails.mp4"
OUTPUT_CSV   = "/home/maaz/eysip/new/trajectories.csv"
OUTPUT_HEATMAP = "/home/maaz/eysip/new/density_heatmap.jpg"  # The final heatmap image will save here
MODEL_PATH   = "/home/maaz/eysip/new/yolov8l.pt"             # yolov8l good for RTX 5060

DETECTION_CLASSES = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck

TRAIL_LEN       = 9999   # effectively infinite — trail stays from first detection
                         # to end of video (no trail cutoff)
SMOOTH_ALPHA    = 0.1    # 0.0 = very smooth but lags, 1.0 = raw jittery positions
                         # 0.25 is good balance for walking/driving speeds
CONF_THRES      = 0.30
IOU_THRES       = 0.45
IMG_SIZE        = 1280

TRAIL_THICKNESS = 1

# Rider fix: if a person's box overlaps more than this fraction
# INSIDE a vehicle box → they're a rider → skip their trail
RIDER_IOA_THRESH = 0.55

# How many frames a track can be missing before we stop updating it
# (trail itself stays drawn — this just stops adding new ghost points)
MAX_TRACK_AGE = 100

# Colors (BGR)
PERSON_COLOR  = (0, 255, 0)     # green
VEHICLE_COLOR = (0,   0,   255) # red
# ══════════════════════════════════════════════════════

DEVICE = "cpu"

# Final-draw smoothing window (frames) — larger = smoother but more lag
SMOOTH_WINDOW = 7


def get_ioa(person_box, vehicle_box):
    """
    Intersection over Person Area.
    Returns what fraction of the person box is inside the vehicle box.
    Used to detect riders — if person is mostly inside a vehicle box = rider.
    """
    px1, py1, px2, py2 = person_box
    vx1, vy1, vx2, vy2 = vehicle_box
    ix1 = max(px1, vx1); iy1 = max(py1, vy1)
    ix2 = min(px2, vx2); iy2 = min(py2, vy2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    person_area = max((px2-px1)*(py2-py1), 1)
    return inter / person_area


def compute_smoothness(trajectory_points):
    """
    Smoothness score: average direction change per step.
    Lower = smoother path (better tracking).
    Higher = jittery path (ID switches or noise).
    Returns angle change in degrees (0=perfectly smooth, 180=zigzag).
    """
    if len(trajectory_points) < 3:
        return 0.0
    pts   = np.array(trajectory_points, dtype=np.float32)
    vecs  = pts[1:] - pts[:-1]
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-6
    vecs_n = vecs / norms
    dots  = np.clip(np.sum(vecs_n[:-1] * vecs_n[1:], axis=1), -1, 1)
    angles = np.degrees(np.arccos(dots))
    return float(np.mean(angles))


def smooth_trail_points(trail, window=7):
    """Apply a simple moving-average filter to a trail (list of (x,y)).
    Returns an (N,2) int numpy array of smoothed points (same length).
    """
    pts = np.array(trail, dtype=np.float32)
    n = len(pts)
    if n < 3 or window <= 1:
        return pts.astype(np.int32)
    w = min(window, n)
    pad = w // 2
    padded = np.pad(pts, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(w, dtype=np.float32) / float(w)
    x = np.convolve(padded[:, 0], kernel, mode="valid")
    y = np.convolve(padded[:, 1], kernel, mode="valid")
    sm = np.vstack((x, y)).T
    return np.round(sm).astype(np.int32)


def generate_density_heatmap(base_frame, trajectories):
    height, width = base_frame.shape[:2]
    heatmap_canvas = np.zeros((height, width), dtype=np.float32)

    # Accumulate tracking points
    for tid, path in trajectories.items():
        for x, y in path:
            x, y = int(x), int(y)
            if 0 <= x < width and 0 <= y < height:
                heatmap_canvas[y, x] += 1 

    # ── THE FIX ──────────────────────────────────────────────
    # Cap the maximum heat accumulation. 
    # If a car sits still and builds up 300 points, we cap it at 15.
    # This ensures moving traffic (e.g., 5-15 overlapping points) 
    # still registers as high heat (red/yellow).
    max_heat_cap = 0.5
    heatmap_canvas = np.clip(heatmap_canvas, 0, max_heat_cap)
    # ─────────────────────────────────────────────────────────

    # Smooth the lines into blobs (slightly smaller kernel for tighter lines)
    heatmap_canvas = cv2.GaussianBlur(heatmap_canvas, (41, 41), 0)

    # Normalize to 0-255 scale
    heatmap_canvas = cv2.normalize(
        heatmap_canvas, None, 
        alpha=0, beta=255, 
        norm_type=cv2.NORM_MINMAX, 
        dtype=cv2.CV_8U
    )

    # Apply the thermal color map
    color_heatmap = cv2.applyColorMap(heatmap_canvas, cv2.COLORMAP_JET)

    # Masking: only overlay colors where traffic actually occurred
    threshold = 2 
    mask = heatmap_canvas > threshold
    
    # Blend with original frame
    alpha = 0.1 
    output_frame = base_frame.copy()
    output_frame[mask] = cv2.addWeighted(base_frame, 1 - alpha, color_heatmap, alpha, 0)[mask]

    return output_frame


def main():
    # ── Open video ──────────────────────────────────────────
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {INPUT_VIDEO}")
        print("        Check INPUT_VIDEO path at top of script.")
        return

    FPS     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    TOTAL   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("=" * 50)
    print("  Trajectory Tracker & Heatmap Generator")
    print("=" * 50)
    print(f"  Input  : {INPUT_VIDEO}")
    print(f"  Output : {OUTPUT_VIDEO}")
    print(f"  CSV    : {OUTPUT_CSV}")
    print(f"  Size   : {frame_w}x{frame_h}  {FPS:.0f}fps  {TOTAL} frames")
    print(f"  Device : {'GPU' if DEVICE == 0 else 'CPU'}\n")

    # ── Output video writer ──────────────────────────────────
    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS, (frame_w, frame_h))

    # ── Load model ───────────────────────────────────────────
    print(f"[MODEL] Loading {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    print(f"[MODEL] Ready.\n")

    # ── Per-track state ──────────────────────────────────────
    full_trajectories = defaultdict(list)
    smooth_centers = {}
    class_votes = defaultdict(list)
    trail_colors = {}
    last_seen = {}
    vehicle_boxes_frame = []
    
    id_switch_count = 0
    prev_frame_centers = {}   
    
    csv_rows = []
    frame_idx = 0
    
    last_processed_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        last_processed_frame = frame.copy() # Store for the heatmap at the end

        # ── Run YOLO + BotSORT ─────────────────────────────
        results = model.track(
            frame,
            persist  = True,
            verbose  = False,
            tracker  = "bytetrack.yaml",
            conf     = CONF_THRES,
            iou      = IOU_THRES,
            imgsz    = IMG_SIZE,
            classes  = DETECTION_CLASSES,
            device   = DEVICE,
        )

        result = results[0]
        boxes  = result.boxes

        # Collect all detections this frame
        detections = []   
        if boxes is not None and boxes.id is not None:
            xyxy      = boxes.xyxy.cpu().numpy()
            track_ids = boxes.id.cpu().numpy().astype(int)
            cls_ids   = boxes.cls.cpu().numpy().astype(int)
            confs     = boxes.conf.cpu().numpy()
            for box, tid, cls_id, conf in zip(xyxy, track_ids, cls_ids, confs):
                x1,y1,x2,y2 = box.astype(int)
                detections.append((x1,y1,x2,y2, int(tid), int(cls_id), float(conf)))

        # ── Rider fix: collect vehicle boxes this frame ────
        vehicle_boxes_frame = [
            (x1,y1,x2,y2)
            for (x1,y1,x2,y2,tid,cls_id,conf) in detections
            if cls_id != 0   
        ]

        # ── Process each detection ─────────────────────────
        current_frame_centers = {}

        for (x1,y1,x2,y2, tid, cls_id, conf) in detections:

            # ── Rider check ───────────────────────────────
            if cls_id == 0:
                is_rider = any(
                    get_ioa((x1,y1,x2,y2), vbox) > RIDER_IOA_THRESH
                    for vbox in vehicle_boxes_frame
                )
                if is_rider:
                    continue   

            # ── Label ────────────────────────────────────
            label = "person" if cls_id == 0 else "vehicle"
            color = PERSON_COLOR if cls_id == 0 else VEHICLE_COLOR
            trail_colors[tid] = color
            class_votes[tid].append(label)
            last_seen[tid] = frame_idx

            # ── Smooth center ─────────────────────────────
            cx = float((x1 + x2) / 2.0)
            cy = float(y2)                  

            raw_pt = np.array([cx, cy], dtype=np.float32)

            if tid not in smooth_centers:
                smooth_centers[tid] = raw_pt
            else:
                smooth_centers[tid] = (
                    SMOOTH_ALPHA * raw_pt +
                    (1.0 - SMOOTH_ALPHA) * smooth_centers[tid]
                )

            smoothed_x = int(round(smooth_centers[tid][0]))
            smoothed_y = int(round(smooth_centers[tid][1]))
            smoothed_pt = (smoothed_x, smoothed_y)

            current_frame_centers[tid] = smoothed_pt

            # ── Store in full trajectory ──────────────────
            full_trajectories[tid].append(smoothed_pt)
            
            # ── CSV row ───────────────────────────────────
            csv_rows.append({
                "frame_id"   : frame_idx,
                "timestamp_s": round(frame_idx / FPS, 3),
                "track_id"   : tid,
                "label"      : label,
                "x"          : smoothed_x,
                "y"          : smoothed_y,
                "raw_x"      : int(cx),
                "raw_y"      : int(cy),
                "conf"       : round(conf, 3),
            })

        # ── ID switch detection (accuracy metric) ─────────
        for prev_tid, prev_pt in prev_frame_centers.items():
            if prev_tid not in current_frame_centers:
                for curr_tid, curr_pt in current_frame_centers.items():
                    if curr_tid not in prev_frame_centers:
                        dist = np.sqrt((curr_pt[0]-prev_pt[0])**2 +
                                       (curr_pt[1]-prev_pt[1])**2)
                        if dist < 100:
                            id_switch_count += 1
                            break
        prev_frame_centers = current_frame_centers

        # ── Draw ALL trajectory lines on clean frame ──────
        overlay = frame.copy()

        for tid, traj in full_trajectories.items():
            if len(traj) < 2:
                continue

            color = trail_colors.get(tid, VEHICLE_COLOR)
            sm_traj = smooth_trail_points(traj, window=SMOOTH_WINDOW)
            pts   = sm_traj.reshape(-1, 1, 2)

            n = len(sm_traj)
            segment_size = max(1, n // 10)

            for seg_start in range(0, n-1, segment_size):
                seg_end = min(seg_start + segment_size + 1, n)
                alpha   = (seg_start / max(n, 1)) * 0.7 + 0.3  
                seg_pts = np.array(sm_traj[seg_start:seg_end],
                                   dtype=np.int32).reshape(-1,1,2)
                faded   = tuple(int(c * alpha) for c in color)
                cv2.polylines(overlay, [seg_pts], False,
                              faded, TRAIL_THICKNESS, cv2.LINE_AA)

            if traj:
                cv2.circle(overlay, tuple(sm_traj[-1]), 4, color, -1)
                cv2.circle(overlay, tuple(sm_traj[-1]), 5, (255,255,255), 1)
            
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

        # ── Minimal HUD — frame counter only ──────────────
        pct = int(frame_idx / max(TOTAL,1) * 100)
        cv2.putText(frame, f"{frame_idx}/{TOTAL}  ({pct}%)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (180,180,180), 1)

        lx = frame_w - 130
        cv2.circle(frame, (lx, 18), 5, PERSON_COLOR, -1)
        cv2.putText(frame, "pedestrian", (lx+10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
        cv2.circle(frame, (lx, 36), 5, VEHICLE_COLOR, -1)
        cv2.putText(frame, "vehicle", (lx+10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

        writer.write(frame)

        # ── Preview ───────────────────────────────────────
        disp = frame
        if frame_w > 1280:
            disp = cv2.resize(frame, (1280, int(frame_h*1280/frame_w)))
        cv2.imshow("Trajectories — ESC to stop", disp)
        if cv2.waitKey(1) & 0xFF == 27:
            print("\n[INFO] Stopped early.")
            break

        frame_idx += 1

        # Progress
        if frame_idx % 30 == 0:
            bar = "█"*(pct//5) + "░"*(20-pct//5)
            print(f"\r  [{bar}] {pct}%  frame {frame_idx}/{TOTAL}", end="")

        # Cleanup live state 
        stale = [t for t,f in last_seen.items() if (frame_idx-f) > MAX_TRACK_AGE]
        for t in stale:
            last_seen.pop(t, None)
            smooth_centers.pop(t, None)
            class_votes.pop(t, None)

    # ── Heatmap Generation (End of Video) ──────────────────
    print("\n\n[INFO] Generating density heatmap...")
    if last_processed_frame is not None:
        heatmap_img = generate_density_heatmap(last_processed_frame, full_trajectories)
        cv2.imwrite(OUTPUT_HEATMAP, heatmap_img)
        print(f"[INFO] Saved heatmap image to: {OUTPUT_HEATMAP}")
        
        disp_hm = heatmap_img
        if frame_w > 1280:
            disp_hm = cv2.resize(heatmap_img, (1280, int(frame_h*1280/frame_w)))
        cv2.imshow("Final Density Heatmap (Press Any Key to close)", disp_hm)
        cv2.waitKey(0) 
    else:
        print("[WARNING] Could not generate heatmap (no frames processed).")

    # ── Write CSV ──────────────────────────────────────────
    with open(OUTPUT_CSV, "w", newline="") as f:
        fieldnames = ["frame_id","timestamp_s","track_id","label",
                      "x","y","raw_x","raw_y","conf"]
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(csv_rows)

    # ── Accuracy metrics ───────────────────────────────────
    smoothness_scores = []
    for tid, traj in full_trajectories.items():
        if len(traj) >= 3:
            smoothness_scores.append(compute_smoothness(traj))

    avg_smoothness = np.mean(smoothness_scores) if smoothness_scores else 0
    total_tracks   = len(full_trajectories)
    person_tracks  = sum(1 for tid in trail_colors if trail_colors[tid] == PERSON_COLOR)
    vehicle_tracks = total_tracks - person_tracks

    # ── Cleanup ────────────────────────────────────────────
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"\n\n{'='*50}")
    print("  Done!")
    print(f"{'='*50}")
    print(f"\n  Output video : {OUTPUT_VIDEO}")
    print(f"  Heatmap image: {OUTPUT_HEATMAP}")
    print(f"  CSV file     : {OUTPUT_CSV}")
    print(f"\n  ── Tracking summary ──")
    print(f"  Total unique objects tracked : {total_tracks}")
    print(f"  Pedestrian tracks            : {person_tracks}")
    print(f"  Vehicle tracks               : {vehicle_tracks}")
    print(f"  Frames processed             : {frame_idx}")
    print(f"\n  ── Accuracy metrics ──")
    print(f"  Avg path smoothness : {avg_smoothness:.1f} degrees")
    print(f"    (lower = smoother = better tracking)")
    print(f"    0-10  = excellent  |  10-25 = good  |  25+ = jittery")
    print(f"  Est. ID switches    : {id_switch_count}")
    print(f"    (lower = better re-identification)")
    print(f"\n  ── How to open CSV ──")
    print(f"  Excel        : double-click {OUTPUT_CSV}")
    print(f"  Google Sheets: sheets.google.com > File > Import > Upload")

if __name__ == "__main__":
    main()