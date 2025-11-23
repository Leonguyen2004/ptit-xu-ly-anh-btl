import os
import cv2
import numpy as np
from tools.detection.license_plate_detector import LicensePlateDetector
from tools.ocr.ocr_model import OCRModel
from tools.image_processing.image_processing import to_grayscale, adjust_brightness_contrast

class LPRPipeline:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.weights_dir = os.path.join(base_dir, 'src', 'weight')
        
        # Paths
        self.yolo_path = os.path.join(self.weights_dir, 'character_detector.pt')
        self.cnn_path = os.path.join(self.weights_dir, 'character_classifier.v2.npz')
        
        # Initialize models
        self.lp_detector = LicensePlateDetector(self.yolo_path)
        self.ocr_model = OCRModel(self.yolo_path, self.cnn_path)

    def run(self, image, mode='accurate'):
        """
        Run the full LPR pipeline.
        Returns:
        {
            'plate_text': str,
            'plate_box': [x1, y1, x2, y2],
            'plate_crop': np.array,
            'characters': list of {box, label, conf, crop},
            'processing_time': float (TODO)
        }
        """
        # 1. Detect License Plate
        lp_detections = self.lp_detector.detect(image)
        
        if not lp_detections:
            return {'error': 'No license plate detected'}
            
        # Take the detection with highest confidence
        best_lp = max(lp_detections, key=lambda x: x['conf'])
        plate_crop = self.lp_detector.crop_plate(image, best_lp)
        
        # 2. OCR on Plate
        text, char_results = self.ocr_model.process_plate(plate_crop, mode=mode)
        
        # Add crops to char results
        for char in char_results:
            x1, y1, x2, y2 = char['box']
            char['crop'] = plate_crop[y1:y2, x1:x2]

        return {
            'plate_text': text,
            'plate_box': best_lp['box'],
            'plate_conf': best_lp['conf'],
            'plate_crop': plate_crop,
            'characters': char_results
        }
