import cv2
import numpy as np
import hashlib
import torch
from collections import defaultdict, Counter

# SAHI & Supervision Imports
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import supervision as sv

# SAM Imports
from segment_anything import sam_model_registry, SamPredictor


# ═══════════════════════════════════════════════════════
#  GLOBAL CONFIGURATION
# ═══════════════════════════════════════════════════════
VIDEO_PATH      = "VID_20251105_180539810.mp4"       # Path to your input video file
OUTPUT_PATH     = "output_locked_ids.mp4" # Path for the processed output video
YOLO_MODEL_PATH = "yolov8x.pt"            # YOLO weight file
SAM_CHECKPOINT  = "sam_vit_h_4b8939.pth"  # SAM weight file ("vit_h", "vit_l", or "vit_b")
SAM_MODEL_TYPE  = "vit_h"                 # Needs to match the downloaded checkpoint

WALKING_CLASSES = {"person"}
RIDER_CLASSES   = {"bicycle", "motorcycle", "motorbike"}
COLOR_WALKING   = (0, 255, 0)
COLOR_DRIVING   = (0, 0, 255)
COLOR_TEXT      = (255, 255, 255)


# ═══════════════════════════════════════════════════════
#  NEW — PERSISTENT ID LOCK REGISTRY (Anti-Flicker)
# ═══════════════════════════════════════════════════════
class PersistentIdLockRegistry:
    def __init__(self, lock_threshold=20, max_coasting_frames=15, iou_threshold=0.40):
        """
        lock_threshold: Frames an object must be active before its ID is locked permanently.
        max_coasting_frames: How many frames to remember a locked target after it disappears.
        iou_threshold: Minimum bounding box overlap to reclaim a fluctuating/switched ID.
        """
        self.lock_threshold = lock_threshold
        self.max_coasting   = max_coasting_frames
        self.iou_threshold  = iou_threshold
        
        self.frame_counters = defaultdict(int)  # tid -> consecutive frames seen
        self.locked_targets = {}                # locked_tid -> {"bbox": (x1,y1,x2,y2), "cls": str, "age": int, "missing": int}

    def compute_iou(self, boxA, boxB):
        """Calculate the Intersection over Union (IoU) of two bounding boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0
        boxAArea = (boxA[2] - boxA[0]) * (boxA[2] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[2] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)

    def process_frame_boxes(self, incoming_boxes):
        """
        Takes raw tracked boxes [(x1, y1, x2, y2, tid, cls, conf), ...]
        Applies anti-fluctuation overrides, and returns the corrected list.
        """
        corrected_boxes = []
        matched_locked_tids = set()

        # Phase 1: Keep track of frame counts for incoming tracker IDs
        for (x1, y1, x2, y2, tid, cls, conf) in incoming_boxes:
            if tid not in self.locked_targets:
                self.frame_counters[tid] += 1
                # If it passes the threshold, promote it to a permanent locked target
                if self.frame_counters[tid] >= self.lock_threshold:
                    self.locked_targets[tid] = {
                        "bbox": (x1, y1, x2, y2),
                        "cls": cls,
                        "missing": 0
                    }

        # Phase 2: Process incoming boxes against our locked target database
        for (x1, y1, x2, y2, tid, cls, conf) in incoming_boxes:
            box = (x1, y1, x2, y2)
            final_tid = tid
            final_cls = cls

            # Case A: The incoming ID is already firmly locked
            if tid in self.locked_targets:
                self.locked_targets[tid]["bbox"] = box
                self.locked_targets[tid]["cls"] = cls
                self.locked_targets[tid]["missing"] = 0
                matched_locked_tids.add(tid)
            
            # Case B: This might be an ID fluctuation (a new/switched ID)
            else:
                best_iou = 0.0
                best_locked_tid = None

                # Look through missing locked targets to see if this box overlaps heavily
                for l_tid, l_data in self.locked_targets.items():
                    if l_data["missing"] > 0 and l_tid not in matched_locked_tids:
                        overlap = self.compute_iou(box, l_data["bbox"])
                        if overlap > best_iou and overlap >= self.iou_threshold:
                            best_iou = overlap
                            best_locked_tid = l_tid

                # Overlap match found: Override back to the historical locked target!
                if best_locked_tid is not None:
                    final_tid = best_locked_tid
                    final_cls = self.locked_targets[best_locked_tid]["cls"]
                    
                    # Update target status with current box properties
                    self.locked_targets[final_tid]["bbox"] = box
                    self.locked_targets[final_tid]["missing"] = 0
                    matched_locked_tids.add(final_tid)

            corrected_boxes.append((x1, y1, x2, y2, final_tid, final_cls, conf))

        # Phase 3: Age out locked targets that have truly left the screen
        expired_tids = []
        for l_tid in self.locked_targets:
            if l_tid not in matched_locked_tids:
                self.locked_targets[l_tid]["missing"] += 1
                if self.locked_targets[l_tid]["missing"] > self.max_coasting:
                    expired_tids.append(l_tid)

        for l_tid in expired_tids:
            del self.locked_targets[l_tid]

        return corrected_boxes


# ═══════════════════════════════════════════════════════
#  STABLE LABEL REGISTRY (Majority Vote Filter)
# ═══════════════════════════════════════════════════════
class StableLabelRegistry:
    def __init__(self, window=15, lock_after=8):
        self.window     = window
        self.lock_after = lock_after
        self.history    = defaultdict(list)
        self.locked     = {}

    def update(self, tid, cls):
        if tid in self.locked:
            return self.locked[tid]
        self.history[tid].append(cls)
        if len(self.history[tid]) > self.window:
            self.history[tid].pop(0)
        winner = Counter(self.history[tid]).most_common(1)[0][0]
        if len(self.history[tid]) >= self.lock_after:
            self.locked[tid] = winner
        return winner


# ═══════════════════════════════════════════════════════
#  GEOMETRIC RIDER RESOLVER
# ═══════════════════════════════════════════════════════
def resolve_rider(tid, cls, box, all_boxes):
    if cls != "person":
        return cls
    x1, y1, x2, y2 = box
    for (bx1, by1, bx2, by2, btid, bcls, _) in all_boxes:
        if bcls not in RIDER_CLASSES or btid == tid:
            continue
        ix1, iy1 = max(x1, bx1), max(y1, by1)
        ix2, iy2 = min(x2, bx2), min(y2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            continue
        union = (x2-x1)*(y2-y1) + (bx2-bx1)*(by2-by1) - inter
        if union > 0 and inter / union > 0.25:
            return "motorcycle"
    return cls


# ═══════════════════════════════════════════════════════
#  PREPROCESSING PIPELINE
# ═══════════════════════════════════════════════════════
def adaptive_clahe(frame, base_clip=2.0):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    variance = cv2.Laplacian(l, cv2.CV_64F).var()
    clip = base_clip if variance > 100 else min(base_clip * 2.5, 6.0)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def motion_aware_sharpen(frame, prev_frame, prev_gray, strength=1.4):
    if prev_gray is None:
        return frame
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    motion_mask = cv2.dilate(motion_mask, np.ones((15, 15), np.uint8), iterations=2)
    blur = cv2.GaussianBlur(frame, (0, 0), 3)
    deblurred = cv2.addWeighted(frame, 1 + strength, blur, -strength, 0)
    m = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    result = (deblurred.astype(np.float32) * m +
              frame.astype(np.float32) * (1 - m))
    return result.astype(np.uint8)

def temporal_blend(prev, curr, alpha=0.2):
    if prev is None:
        return curr
    return cv2.addWeighted(curr, 1 - alpha, prev, alpha, 0)

def preprocess(frame, prev_frame, prev_gray):
    frame = temporal_blend(prev_frame, frame, alpha=0.2)
    frame = adaptive_clahe(frame)
    frame = motion_aware_sharpen(frame, prev_frame, prev_gray, strength=1.4)
    return frame


# ═══════════════════════════════════════════════════════
#  DRAWING HELPERS
# ═══════════════════════════════════════════════════════
def get_color_by_id(tid):
    h = hashlib.md5(str(tid).encode()).hexdigest()
    return (max(int(h[4:6],16),80), max(int(h[2:4],16),80), max(int(h[0:2],16),80))

def overlay_mask(frame, mask, color, alpha=0.40):
    colored = np.zeros_like(frame, dtype=np.uint8)
    colored[mask] = color
    return cv2.addWeighted(frame, 1.0, colored, alpha, 0)


# ═══════════════════════════════════════════════════════
#  INITIALIZATION
# ═══════════════════════════════════════════════════════
device =  "cpu"
print(f"Using execution device: {device}")

# 1. Initialize SAHI Sliced Inference Wrapper
sahi_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=YOLO_MODEL_PATH,
    confidence_threshold=0.35,
    device=device
)

# 2. Initialize SAM Model
sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=device)
predictor = SamPredictor(sam)

# 3. Video Setup
cap   = cv2.VideoCapture(VIDEO_PATH)
W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS   = cap.get(cv2.CAP_PROP_FPS) or 30
TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

# 4. Initialize Standalone ByteTrack Tracker
tracker = sv.ByteTrack(
    track_activation_threshold=0.35,
    minimum_matching_threshold=0.8,
    frame_rate=FPS
)

# 5. Initialize Registries (Anti-Fluctuation Lock replaces Kinematic Classifier)
id_lock_registry = PersistentIdLockRegistry(lock_threshold=20, max_coasting_frames=15, iou_threshold=0.40)
stable_label_registry = StableLabelRegistry(window=15, lock_after=8)

prev_frame = None
prev_gray  = None
frame_idx  = 0


# ═══════════════════════════════════════════════════════
#  MAIN PROCESSING LOOP
# ═══════════════════════════════════════════════════════
while True:
    ret, raw_frame = cap.read()
    if not ret:
        break

    # Preprocess
    frame = preprocess(raw_frame, prev_frame, prev_gray)
    curr_gray_raw = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)

    # SAHI Sliced Inference
    sahi_result = get_sliced_prediction(
        frame,
        sahi_model,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        verbose=0
    )

    # Tracker Update
    # ── Convert SAHI Predictions to Supervision Detections ──
    xyxy = []
    confidence = []
    class_id = []

    # Manually extract the data from SAHI's output
    for pred in sahi_result.object_prediction_list:
        xyxy.append(pred.bbox.to_xyxy())
        confidence.append(pred.score.value)
        class_id.append(pred.category.id)

    # Create the supervision Detections object directly
    if len(xyxy) > 0:
        detections = sv.Detections(
            xyxy=np.array(xyxy),
            confidence=np.array(confidence),
            class_id=np.array(class_id).astype(int)
        )
    else:
        detections = sv.Detections.empty()

    # ── Update Tracker ──
    tracked_detections = tracker.update_with_detections(detections)

    # Parse Frame Elements
    boxes_info = []
    for i in range(len(tracked_detections)):
        xyxy     = tracked_detections.xyxy[i]
        class_id = tracked_detections.class_id[i]
        conf     = float(tracked_detections.confidence[i])
        tid      = int(tracked_detections.tracker_id[i])
        
        x1, y1, x2, y2 = map(int, xyxy)
        cls = sahi_model.category_mapping.get(str(class_id), sahi_model.category_mapping.get(class_id, "unknown"))
        boxes_info.append((x1, y1, x2, y2, tid, cls, conf))

    # STAGE 1: Geometric Rider Overlap Resolver
    resolved_boxes = []
    for (x1, y1, x2, y2, tid, cls, conf) in boxes_info:
        cls = resolve_rider(tid, cls, (x1, y1, x2, y2), boxes_info)
        resolved_boxes.append((x1, y1, x2, y2, tid, cls, conf))

    # STAGE 2: Anti-Fluctuation Persistent ID Lock (Replaces Kinematics)
    locked_boxes = id_lock_registry.process_frame_boxes(resolved_boxes)

    # STAGE 3: Majority-Vote Label Stabilization Lock
    final_boxes = []
    for (x1, y1, x2, y2, tid, cls, conf) in locked_boxes:
        stable_cls = stable_label_registry.update(tid, cls)
        final_boxes.append((x1, y1, x2, y2, tid, stable_cls, conf))

    # SAM Pixel-Perfect Segmentation Engine
    if final_boxes:
        predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        for (x1, y1, x2, y2, tid, cls, conf) in final_boxes:
            is_walking = cls.lower() in WALKING_CLASSES
            label_text = "WALKING" if is_walking else "DRIVING"
            box_color  = COLOR_WALKING if is_walking else COLOR_DRIVING
            id_color   = get_color_by_id(tid)

            input_box = np.array([[x1, y1, x2, y2]])

            try:
                masks, scores, _ = predictor.predict(box=input_box, multimask_output=False)
                best_mask = masks[0].astype(bool)
                frame = overlay_mask(frame, best_mask, box_color, alpha=0.40)
                contours, _ = cv2.findContours(best_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, box_color, 2)
            except Exception:
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            label = f"ID:{tid} {cls} | {label_text}"
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1-th-bl-6), (x1+tw+6, y1), id_color, -1)
            cv2.putText(frame, label, (x1+3, y1-bl-3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)

    # Legend UI Render
    cv2.rectangle(frame, (8, 8), (220, 58), (30, 30, 30), -1)
    cv2.putText(frame, "WALKING", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WALKING, 2)
    cv2.putText(frame, "DRIVING", (120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_DRIVING, 2)
    cv2.putText(frame, f"Frame {frame_idx}/{TOTAL}", (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    out.write(frame)

    # Update State Reference Keys
    prev_frame = raw_frame.copy()
    prev_gray  = curr_gray_raw
    frame_idx += 1

    if frame_idx % 5 == 0:
        pct = (frame_idx / TOTAL * 100) if TOTAL > 0 else 0
        bar = "█" * int(pct//5) + "░" * (20 - int(pct//5))
        print(f"\r[{bar}] {pct:.1f}%  frame {frame_idx}/{TOTAL}", end="")

cap.release()
out.release()
print(f"\n✅ Processing Complete! Output saved → {OUTPUT_PATH}")