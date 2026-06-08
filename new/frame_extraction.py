import os
import cv2
import random

def extract_frames(video_path, output_dir, num_frames_to_extract=30, method="even"):
    """
    Extracts frames from a video file for annotation.
    
    Args:
        video_path (str): Path to the source video file.
        output_dir (str): Directory where extracted frames will be saved.
        num_frames_to_extract (int): Number of frames to sample.
        method (str): 'even' for uniform spacing, 'random' for random sampling.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")

    # Determine frame indices to extract
    if method == "even":
        frame_indices = [int(i * total_frames / num_frames_to_extract) for i in range(num_frames_to_extract)]
    elif method == "random":
        frame_indices = sorted(random.sample(range(total_frames), min(num_frames_to_extract, total_frames)))
    else:
        raise ValueError("Method must be either 'even' or 'random'")

    saved_count = 0
    for idx in frame_indices:
        # Set video reader pointer to the specific frame index
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        
        if success:
            frame_name = f"frame_{idx:06d}.jpg"
            out_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(out_path, frame)
            saved_count += 1
        else:
            print(f"Warning: Failed to extract frame at index {idx}")

    cap.release()
    print(f"Successfully extracted and saved {saved_count} frames to '{output_dir}'.")

# Example Usage
if __name__ == "__main__":
    VIDEO_FILE = "new/input.mp4"  # Replace with your video path
    OUTPUT_FOLDER = "extracted_images"
    
    extract_frames(video_path=VIDEO_FILE, output_dir=OUTPUT_FOLDER, num_frames_to_extract=200, method="even")