from ultralytics import YOLO
import cv2
import numpy as np
import os

class LicensePlateDetector:
    def __init__(self, model_path):
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        self.model = YOLO(model_path)
        # Class 31 is 'license plates' in the provided model
        self.lp_class_id = 31 

    def detect(self, image, conf_threshold=0.25):
        """
        Detect license plates in the image.
        If class 31 (license plate) is not found, infer it from character detections.
        Returns a list of detections: {'box': [x1, y1, x2, y2], 'conf': float, 'class_id': int}
        """
        results = self.model(image, conf=conf_threshold, verbose=False)
        detections = []
        character_boxes = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0].cpu().numpy())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                if cls_id == self.lp_class_id:
                    detections.append({
                        'box': [int(x1), int(y1), int(x2), int(y2)],
                        'conf': conf,
                        'class_id': cls_id
                    })
                else:
                    # It's a character
                    character_boxes.append([x1, y1, x2, y2])

        # Fallback: If no plate detected but characters are found, 
        # infer plate box from characters
        if not detections and character_boxes:
            character_boxes = np.array(character_boxes)
            min_x = np.min(character_boxes[:, 0])
            min_y = np.min(character_boxes[:, 1])
            max_x = np.max(character_boxes[:, 2])
            max_y = np.max(character_boxes[:, 3])
            
            # Add padding (e.g., 10%)
            w = max_x - min_x
            h = max_y - min_y
            pad_x = w * 0.1
            pad_y = h * 0.1
            
            img_h, img_w = image.shape[:2]
            x1 = max(0, int(min_x - pad_x))
            y1 = max(0, int(min_y - pad_y))
            x2 = min(img_w, int(max_x + pad_x))
            y2 = min(img_h, int(max_y + pad_y))
            
            detections.append({
                'box': [x1, y1, x2, y2],
                'conf': 0.5, # Dummy confidence
                'class_id': self.lp_class_id
            })
            
        return detections

    def crop_plate(self, image, detection):
        """Crops the license plate from the image based on detection."""
        x1, y1, x2, y2 = detection['box']
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return image[y1:y2, x1:x2]
