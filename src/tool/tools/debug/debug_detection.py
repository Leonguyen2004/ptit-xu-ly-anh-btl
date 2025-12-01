import sys
import os

# Add project root to sys.path (go up 5 levels: debug -> tools -> tool -> src -> root)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from ..detection.license_plate_detector import LicensePlateDetector
import cv2

# Paths
# BASE_DIR should point to project root (go up 5 levels)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
MODEL_PATH = os.path.join(BASE_DIR, "src", "weight", "character_detector.pt")
IMAGE_PATH = os.path.join(BASE_DIR, "src", "character_cropper", "images", "car_2.jpg")

def debug_detection():
    print(f"Loading model from {MODEL_PATH}")
    detector = LicensePlateDetector(MODEL_PATH)
    
    print(f"Loading image from {IMAGE_PATH}")
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("Failed to load image")
        return

    print("Running detection...")
    detections = detector.detect(img, conf_threshold=0.1)
    
    print(f"Found {len(detections)} plate detections:")
    for i, det in enumerate(detections):
        print(f"Detection {i}: {det}")
        
        # Draw box
        x1, y1, x2, y2 = det['box']
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
    # Save debug image
    out_path = os.path.join(BASE_DIR, "debug_output.jpg")
    cv2.imwrite(out_path, img)
    print(f"Saved debug image to {out_path}")

if __name__ == "__main__":
    debug_detection()
