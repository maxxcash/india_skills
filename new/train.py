import os
import torch
from ultralytics import YOLO

def main():
    # 1. Check for GPU acceleration
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"--- Training Execution Device: {device} ---")
    if device == "0":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 2. Point to your fixed data.yaml
    yaml_path = "/home/maaz/eysip/new/traffic analysis.yolo26/data.yaml"
    
    if not os.path.exists(yaml_path):
        print(f"Error: Cannot find data.yaml at {yaml_path}")
        return

    # 3. Load YOLO26x weights (Extra-Large variant)
    print("Loading YOLO26x model parameters...")
    model = YOLO("yolo26x-seg.pt")

    # 4. Fire up the training process
    print("Starting training loop...")
    model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=8, # Drop this to 4 if the 'x' model causes a CUDA Out of Memory error
        device=device,
        workers=4,
        project="/home/maaz/eysip/new/Traffic_Project",
        name="yolo26x_traffic"
    )

if __name__ == "__main__":
    main()