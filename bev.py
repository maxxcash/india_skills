import cv2
import numpy as np
from tqdm import tqdm

# 1. Setup Video Input and Output
video_path = 'VID_20251105_180539810.mp4'
output_bev_video = 'BEV_Transformed_Video.mp4'

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

if not cap.isOpened():
    print("Error: Could not open the video.")
    exit()

# 2. Define the Transformation Matrix
# Your verified perspective coordinates
src_pts = np.float32([[754, 1], [12, 426], [1802, 1076], [1650, 11]])

# Destination coordinates (This dictates the resolution of your new video)
# A 1000x1000 square grid is a good starting point for a flat map
BEV_WIDTH = 1000
BEV_HEIGHT = 1000
dst_pts = np.float32([
    [0, 0], 
    [BEV_WIDTH, 0], 
    [BEV_WIDTH, BEV_HEIGHT], 
    [0, BEV_HEIGHT]
])

matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

# 3. Setup the Video Writer with the NEW dimensions
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_bev_video, fourcc, fps, (BEV_WIDTH, BEV_HEIGHT))

print("Starting Bird's-Eye View video conversion...")
pbar = tqdm(total=total_frames, desc="Warping Frames", unit="frame")

# 4. Processing Loop
while True:
    success, frame = cap.read()
    if not success:
        break
        
    # Warp the entire image frame using the matrix
    warped_frame = cv2.warpPerspective(frame, matrix, (BEV_WIDTH, BEV_HEIGHT))
    
    # Write the transformed frame to the new video file
    out.write(warped_frame)
    pbar.update(1)

# 5. Clean up
pbar.close()
cap.release()
out.release()
print(f"Done! Saved top-down video as: {output_bev_video}")