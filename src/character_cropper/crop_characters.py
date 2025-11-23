import os
import cv2
from ultralytics import YOLO
from pathlib import Path
import torch
from ultralytics.nn.tasks import DetectionModel

torch.serialization.add_safe_globals([DetectionModel])

def crop_characters():
    # Define paths
    base_dir = Path(__file__).parent
    # Model is located at src/weight/character_detector.pt
    model_path = base_dir.parent / "weight" / "character_detector.pt"
    images_dir = base_dir / "images"
    output_dir = base_dir / "out_chars"

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if model exists
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    # Load the model
    print(f"Loading model from {model_path}...")
    model = YOLO(str(model_path))

    # Process images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = [f for f in os.listdir(images_dir) if Path(f).suffix.lower() in image_extensions]

    if not image_files:
        print(f"No images found in {images_dir}")
        return

    print(f"Found {len(image_files)} images. Processing...")

    for img_file in image_files:
        img_path = images_dir / img_file
        print(f"Processing {img_file}...")
        
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Could not read image {img_path}")
            continue

        # Run inference
        results = model(img)

        # Process results
        for r in results:
            boxes = r.boxes
            print(f"  Found {len(boxes)} detections")
            for i, box in enumerate(boxes):
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Crop the character
                # Ensure coordinates are within image bounds
                h, w = img.shape[:2]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                crop = img[y1:y2, x1:x2]
                
                if crop.size == 0:
                    continue

                # Get class name
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                
                # Create class directory
                class_dir = output_dir / cls_name
                class_dir.mkdir(parents=True, exist_ok=True)

                # Save the cropped image
                # Naming convention: {original_filename}_char_{index}.jpg
                # Remove extension from original filename for the prefix
                stem = Path(img_file).stem
                out_name = f"{stem}_char_{i}.jpg"
                out_path = class_dir / out_name
                
                cv2.imwrite(str(out_path), crop)
                print(f"  Saved {out_name} to {cls_name}/")

    print("Done processing images.")

if __name__ == "__main__":
    crop_characters()
