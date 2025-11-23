import os
import sys
import numpy as np
import cv2
from PIL import Image as PILImage
from ultralytics import YOLO

# Add cnn directory to path to import CNNModel
# Get project root (2 levels up from tools/ocr/)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
cnn_dir = os.path.join(project_root, 'src', 'cnn')
if cnn_dir not in sys.path:
    sys.path.append(cnn_dir)

try:
    from model import CNNModel
except ImportError:
    # Fallback or handle error if model.py is not found
    print("Warning: Could not import CNNModel from src/cnn/model.py")
    CNNModel = None

class OCRModel:
    def __init__(self, yolo_path, cnn_path=None):
        self.yolo_path = yolo_path
        self.cnn_path = cnn_path
        
        # Load YOLO model
        if not os.path.exists(yolo_path):
            raise FileNotFoundError(f"YOLO model not found at {yolo_path}")
        self.yolo_model = YOLO(yolo_path)
        
        # Load CNN model if path provided
        self.cnn_model = None
        self.idx_to_class = None
        if cnn_path and CNNModel:
            if os.path.exists(cnn_path):
                self.cnn_model, self.idx_to_class = CNNModel.load_model(cnn_path)
                if self.cnn_model is None:
                    print(f"Failed to load CNN model from {cnn_path}")
            else:
                print(f"CNN model file not found at {cnn_path}")

    def detect_characters(self, plate_image, conf_threshold=0.25):
        """
        Detect characters in the license plate image using YOLO.
        Returns list of {'box': [x1,y1,x2,y2], 'class_id': int, 'conf': float, 'label': str}
        """
        results = self.yolo_model(plate_image, conf=conf_threshold, verbose=False)
        detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                # Filter out 'license plate' class (id 31)
                if cls_id == 31:
                    continue
                    
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                label = self.yolo_model.names[cls_id]
                
                detections.append({
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'conf': conf,
                    'class_id': cls_id,
                    'label': label
                })
        
        # Sort detections for 2-row license plates (Vietnamese format)
        # Group characters by row (top/bottom) based on y-coordinate
        if len(detections) > 0:
            # Calculate average y-coordinate
            avg_y = sum(det['box'][1] + det['box'][3] for det in detections) / (2 * len(detections))
            
            # Separate into top and bottom rows
            top_row = []
            bottom_row = []
            
            for det in detections:
                y_center = (det['box'][1] + det['box'][3]) / 2
                if y_center < avg_y:
                    top_row.append(det)
                else:
                    bottom_row.append(det)
            
            # Sort each row from left to right
            top_row.sort(key=lambda x: x['box'][0])
            bottom_row.sort(key=lambda x: x['box'][0])
            
            # Combine: top row first, then bottom row
            detections = top_row + bottom_row
        
        return detections

    def recognize_char_cnn(self, char_image):
        """
        Recognize a single character image using the CNN model.
        """
        if self.cnn_model is None:
            return None, 0.0
            
        try:
            # Preprocess image for CNN (28x28, Grayscale, Normalized)
            if len(char_image.shape) == 3:
                gray = cv2.cvtColor(char_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = char_image
                
            img_pil = PILImage.fromarray(gray)
            img_resized = img_pil.resize((28, 28), PILImage.LANCZOS)
            img_array = np.array(img_resized, dtype=np.float32) / 255.0
            
            # Forward pass
            probs, _ = self.cnn_model.forward(img_array, training=False)
            pred_idx = np.argmax(probs)
            pred_label = self.idx_to_class[pred_idx]
            confidence = probs[pred_idx]
            
            return pred_label, confidence
        except Exception as e:
            print(f"Error in CNN recognition: {e}")
            return None, 0.0

    def process_plate(self, plate_image, mode='fast'):
        """
        Full process: Detect characters -> Recognize (YOLO or CNN) -> Return Text
        mode: 'fast' (YOLO only) or 'accurate' (YOLO detect + CNN recognize)
        """
        detections = self.detect_characters(plate_image)
        
        results = []
        full_text = ""
        
        for det in detections:
            x1, y1, x2, y2 = det['box']
            char_crop = plate_image[y1:y2, x1:x2]
            
            if mode == 'accurate' and self.cnn_model:
                label, conf = self.recognize_char_cnn(char_crop)
                if label:
                    det['label'] = label
                    det['conf'] = float(conf) # Update confidence from CNN
            
            results.append(det)
            full_text += det['label']
            
        return full_text, results
