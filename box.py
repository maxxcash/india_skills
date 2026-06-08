import cv2
import numpy as np

video_path = 'VID_20251105_180539810.mp4'
points = []

def select_point(event, x, y, flags, param):
    global points, frame_copy
    # Record the point on left mouse click
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append([x, y])
            cv2.circle(frame_copy, (x, y), 5, (0, 255, 0), -1)
            
            # If it's the 4th point, draw the final shape and print the array
            if len(points) == 4:
                cv2.polylines(frame_copy, [np.array(points)], True, (0, 0, 255), 2)
                print("\n--- Copy this into your Colab Script ---")
                print(f"src_pts = np.float32({points})")
                print("----------------------------------------\n")
            
            cv2.imshow("Click 4 points. Press 'q' to quit.", frame_copy)

# 1. Extract the first frame
cap = cv2.VideoCapture(video_path)
success, frame = cap.read()
cap.release()

if not success:
    print("Could not read video.")
    exit()

frame_copy = frame.copy()

# 2. Open the interactive window
cv2.namedWindow("Click 4 points. Press 'q' to quit.", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Click 4 points. Press 'q' to quit.", select_point)
cv2.imshow("Click 4 points. Press 'q' to quit.", frame_copy)

# Wait until 'q' is pressed
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()