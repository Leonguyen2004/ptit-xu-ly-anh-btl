import os
import sys
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import base64
from pathlib import Path
from PIL import Image

# Add src to path to import CNN model
# tools is now in src/tools, so parent is src
current_dir = Path(__file__).parent.absolute()
src_dir = current_dir.parent
cnn_dir = src_dir / "cnn"

if str(cnn_dir) not in sys.path:
    sys.path.append(str(cnn_dir))

from model import CNNModel

class LicensePlateRecognizer:
    def __init__(self):
        self.yolo_model = None
        self.cnn_model = None
        self.idx_to_class = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.load_models()

    def load_models(self):
        """Load both YOLO and CNN models"""
        # Load YOLO
        yolo_path = src_dir / "weight" / "character_detector.pt"
        if yolo_path.exists():
            self.yolo_model = YOLO(str(yolo_path))
        else:
            print(f"Warning: YOLO model not found at {yolo_path}")

        # Load CNN
        cnn_path = src_dir / "weight" / "character_classifier.v2.npz"
        if cnn_path.exists():
            self.cnn_model, self.idx_to_class = CNNModel.load_model(str(cnn_path))
        else:
            print(f"Warning: CNN model not found at {cnn_path}")

    def preprocess_character(self, img_crop):
        """Resize and normalize character crop for CNN"""
        # Convert to grayscale if not already
        if len(img_crop.shape) == 3:
            img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img_crop

        # Resize to 28x28
        img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_LANCZOS4)
        
        # Normalize to 0-1
        img_norm = img_resized.astype(np.float32) / 255.0
        
        return img_norm

    def process_image(self, image_pil, yolo_conf=0.25):
        """
        Process the full pipeline:
        1. Detect characters with YOLO
        2. Sort characters by position
        3. Crop and classify each character
        """
        if self.yolo_model is None or self.cnn_model is None:
            return None, "Models not loaded correctly"

        # Convert PIL to cv2
        img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        # 1. YOLO Detection
        results = self.yolo_model(img_cv, conf=yolo_conf)[0]
        
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            detections.append({
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'conf': conf
            })

        # 2. Sort detections left-to-right, top-to-bottom
        # Simple sorting by x1 might fail for two-line plates.
        # Heuristic: Sort by Y first (with tolerance), then X.
        def sort_key(d):
            x1, y1, x2, y2 = d['bbox']
            return (y1 // 20, x1) # Group by lines of 20px height roughly
            
        detections.sort(key=sort_key)

        recognized_text = ""
        annotated_img = img_cv.copy()
        debug_crops = []

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Crop
            # Add small padding?
            h, w = img_cv.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            crop = img_cv[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Preprocess
            input_tensor = self.preprocess_character(crop)
            
            # Classify
            probs, _ = self.cnn_model.forward(input_tensor, training=False)
            pred_idx = np.argmax(probs)
            char = self.idx_to_class[pred_idx]
            conf = probs[pred_idx]

            recognized_text += char
            
            # Annotate
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_img, f"{char}", (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            debug_crops.append({
                'image': crop,
                'char': char,
                'conf': conf
            })

        # Convert annotated image back to RGB for Streamlit/Flask
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        # Convert images to base64 for JSON response
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(annotated_img_rgb, cv2.COLOR_RGB2BGR))
        annotated_img_b64 = base64.b64encode(buffer).decode('utf-8')

        debug_crops_data = []
        for item in debug_crops:
            _, buf = cv2.imencode('.jpg', item['image'])
            b64 = base64.b64encode(buf).decode('utf-8')
            debug_crops_data.append({
                'image': b64,
                'char': item['char'],
                'conf': float(item['conf'])
            })
        
        return {
            'text': recognized_text,
            'annotated_image': annotated_img_b64,
            'debug_crops': debug_crops_data
        }

    def analyze_image(self, image_pil):
        """
        Perform image analysis steps for visualization:
        1. Grayscale
        2. Gaussian Blur
        3. Canny Edges
        4. Histogram Equalization (Enhancement)
        5. Otsu Thresholding (Segmentation)
        6. Morphology (Dilation)
        """
        img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        # 1. Grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # 2. Gaussian Blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Canny Edges
        edges = cv2.Canny(blur, 50, 150)
        
        # 4. Histogram Equalization
        equ = cv2.equalizeHist(gray)
        
        # 5. Otsu Thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 6. Morphology (Dilation)
        kernel = np.ones((3,3), np.uint8)
        dilation = cv2.dilate(thresh, kernel, iterations=1)
        
        # Helper to convert to base64
        def to_b64(img):
            _, buf = cv2.imencode('.jpg', img)
            return base64.b64encode(buf).decode('utf-8')

        return {
            'original': to_b64(img_cv),
            'grayscale': to_b64(gray),
            'blur': to_b64(blur),
            'edges': to_b64(edges),
            'equalized': to_b64(equ),
            'threshold': to_b64(thresh),
            'morphology': to_b64(dilation)
        }
