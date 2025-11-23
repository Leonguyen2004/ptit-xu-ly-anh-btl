import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image as PILImage

# Add cnn directory to path to import CNNModel
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
cnn_dir = os.path.join(project_root, 'src', 'cnn')
if cnn_dir not in sys.path:
    sys.path.append(cnn_dir)

from model import CNNModel


class LeoPipeline:
    """
    Pipeline xử lý nhận diện biển số xe:
    1. Sử dụng character_detector.pt để detect và crop các ký tự
    2. Sắp xếp ký tự theo thứ tự từ trái sang phải, từ trên xuống dưới
    3. Sử dụng character_classifier.v2.npz để phân loại từng ký tự
    """
    
    def __init__(self, base_dir):
        """
        Khởi tạo pipeline với đường dẫn đến thư mục gốc của project.
        
        Args:
            base_dir: Đường dẫn đến thư mục gốc của project
        """
        self.base_dir = base_dir
        self.weights_dir = os.path.join(base_dir, 'src', 'weight')
        
        # Đường dẫn đến các model
        self.character_detector_path = os.path.join(self.weights_dir, 'character_detector.pt')
        self.character_classifier_path = os.path.join(self.weights_dir, 'character_classifier.v2.npz')
        
        # Load models
        self.yolo_model = None
        self.cnn_model = None
        self.idx_to_class = None
        
        self._load_models()
    
    def _load_models(self):
        """Load YOLO detector và CNN classifier models"""
        # Load YOLO model
        if not os.path.exists(self.character_detector_path):
            raise FileNotFoundError(f"Character detector not found at {self.character_detector_path}")
        self.yolo_model = YOLO(self.character_detector_path)
        
        # Load CNN model
        if not os.path.exists(self.character_classifier_path):
            raise FileNotFoundError(f"Character classifier not found at {self.character_classifier_path}")
        
        self.cnn_model, self.idx_to_class = CNNModel.load_model(self.character_classifier_path)
        if self.cnn_model is None:
            raise RuntimeError(f"Failed to load CNN model from {self.character_classifier_path}")
    
    def calculate_iou(self, box1, box2):
        """
        Tính IoU (Intersection over Union) giữa 2 bounding boxes.
        
        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]
        
        Returns:
            float: IoU value (0.0 - 1.0)
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Tính diện tích intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Tính diện tích union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def remove_duplicate_detections(self, detections, iou_threshold=0.5):
        """
        Loại bỏ các detection trùng lặp dựa trên IoU.
        Nếu 2 boxes có IoU > threshold, chỉ giữ lại box có confidence cao hơn.
        
        Args:
            detections: List các detection với format {'box': [x1, y1, x2, y2], 'conf': float, ...}
            iou_threshold: Ngưỡng IoU để coi là trùng lặp (mặc định 0.5)
        
        Returns:
            List các detection đã loại bỏ trùng lặp
        """
        if not detections or len(detections) <= 1:
            return detections
        
        # Sắp xếp theo confidence giảm dần để ưu tiên giữ lại box có confidence cao hơn
        sorted_detections = sorted(detections, key=lambda x: x['conf'], reverse=True)
        
        filtered_detections = []
        keep_flags = [True] * len(sorted_detections)
        
        for i in range(len(sorted_detections)):
            if not keep_flags[i]:
                continue
            
            current_box = sorted_detections[i]['box']
            filtered_detections.append(sorted_detections[i])
            
            # Kiểm tra các box còn lại
            for j in range(i + 1, len(sorted_detections)):
                if not keep_flags[j]:
                    continue
                
                other_box = sorted_detections[j]['box']
                iou = self.calculate_iou(current_box, other_box)
                
                # Nếu IoU > threshold, loại bỏ box có confidence thấp hơn
                if iou > iou_threshold:
                    keep_flags[j] = False
        
        return filtered_detections
    
    def sort_characters_left_to_right_top_to_bottom(self, detections):
        """
        Sắp xếp các ký tự theo thứ tự từ trái sang phải, từ trên xuống dưới.
        Xử lý cả trường hợp biển số 1 hàng và 2 hàng.
        
        Args:
            detections: List các detection với format {'box': [x1, y1, x2, y2], ...}
        
        Returns:
            List các detection đã được sắp xếp
        """
        if not detections:
            return []
        
        # Tính toán trung bình y-coordinate để phân biệt hàng trên và hàng dưới
        avg_y = sum((det['box'][1] + det['box'][3]) / 2 for det in detections) / len(detections)
        
        # Phân loại thành hàng trên và hàng dưới
        top_row = []
        bottom_row = []
        
        for det in detections:
            y_center = (det['box'][1] + det['box'][3]) / 2
            if y_center < avg_y:
                top_row.append(det)
            else:
                bottom_row.append(det)
        
        # Sắp xếp mỗi hàng từ trái sang phải (theo x-coordinate)
        top_row.sort(key=lambda x: x['box'][0])
        bottom_row.sort(key=lambda x: x['box'][0])
        
        # Kết hợp: hàng trên trước, sau đó hàng dưới
        return top_row + bottom_row
    
    def recognize_character_cnn(self, char_image):
        """
        Phân loại ký tự sử dụng CNN model.
        
        Args:
            char_image: Ảnh ký tự (numpy array)
        
        Returns:
            tuple: (label, confidence) hoặc (None, 0.0) nếu lỗi
        """
        if self.cnn_model is None:
            return None, 0.0
        
        try:
            # Preprocess image cho CNN (28x28, Grayscale, Normalized)
            if len(char_image.shape) == 3:
                gray = cv2.cvtColor(char_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = char_image
            
            # Resize về 28x28 và normalize
            img_pil = PILImage.fromarray(gray)
            img_resized = img_pil.resize((28, 28), PILImage.LANCZOS)
            img_array = np.array(img_resized, dtype=np.float32) / 255.0
            
            # Forward pass qua CNN model
            probs, _ = self.cnn_model.forward(img_array, training=False)
            pred_idx = np.argmax(probs)
            pred_label = self.idx_to_class[pred_idx]
            confidence = float(probs[pred_idx])
            
            return pred_label, confidence
        except Exception as e:
            print(f"Error in CNN recognition: {e}")
            return None, 0.0
    
    def process(self, image, encode_image_func):
        """
        Xử lý ảnh để nhận diện các ký tự trên biển số.
        
        Args:
            image: Ảnh input (numpy array)
            encode_image_func: Hàm để encode ảnh thành base64 string
        
        Returns:
            dict: Kết quả xử lý với các keys:
                - plate_text: Chuỗi ký tự đã nhận diện
                - plate_conf: Độ tin cậy trung bình
                - result_image: Ảnh gốc với bounding boxes quanh các ký tự (base64) - để hiển thị ở ô result
                - visualization: Ảnh với bounding boxes và labels (base64) - để debug/chi tiết
                - characters: List các ký tự với thông tin chi tiết
        """
        # 1. Sử dụng character_detector.pt để detect và crop các ký tự
        results = self.yolo_model(image, conf=0.25, iou=0.5, verbose=False)
        
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                # Bỏ qua class license plate (id 31) nếu có
                if cls_id == 31:
                    continue
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = float(box.conf[0].cpu().numpy())
                
                # Đảm bảo tọa độ nằm trong phạm vi ảnh
                h, w = image.shape[:2]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                # Crop ký tự
                char_crop = image[y1:y2, x1:x2]
                
                if char_crop.size == 0:
                    continue
                
                detections.append({
                    'box': [x1, y1, x2, y2],
                    'conf': conf,
                    'crop': char_crop
                })
        
        if not detections:
            return {'error': 'No characters detected'}
        
        # 2. Loại bỏ các detection trùng lặp (nếu model detect cùng 1 ký tự 2 lần)
        filtered_detections = self.remove_duplicate_detections(detections, iou_threshold=0.5)
        
        if not filtered_detections:
            return {'error': 'No valid characters after filtering'}
        
        # 3. Sắp xếp các ký tự theo thứ tự từ trái sang phải, từ trên xuống dưới
        sorted_detections = self.sort_characters_left_to_right_top_to_bottom(filtered_detections)
        
        # 4. Sử dụng character_classifier.v2.npz để phân loại từng ký tự
        characters = []
        plate_text = ""
        
        for det in sorted_detections:
            # Phân loại ký tự bằng CNN model
            label, cnn_conf = self.recognize_character_cnn(det['crop'])
            
            if label is None:
                # Nếu CNN không nhận diện được, đánh dấu là "?"
                label = "?"
                cnn_conf = 0.0
            
            characters.append({
                'label': label,
                'conf': float(cnn_conf),
                'box': det['box'],
                'crop': encode_image_func(det['crop'])
            })
            
            plate_text += label
        
        # 5. Tạo ảnh result: ảnh gốc với bounding boxes quanh các ký tự (để hiển thị ở ô result)
        img_result = image.copy()
        for char in characters:
            x1, y1, x2, y2 = char['box']
            # Vẽ bounding box với màu xanh lá, độ dày 2
            cv2.rectangle(img_result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 6. Tạo ảnh visualization: ảnh với bounding boxes và labels (để debug/chi tiết)
        img_vis = image.copy()
        for char in characters:
            x1, y1, x2, y2 = char['box']
            # Vẽ bounding box
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Vẽ label
            cv2.putText(img_vis, char['label'], (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Format response
        result = {
            'plate_text': plate_text,
            'plate_conf': float(np.mean([char['conf'] for char in characters])) if characters else 0.0,
            'result_image': encode_image_func(img_result),  # Ảnh gốc với bounding boxes (để hiển thị ở ô result)
            'visualization': encode_image_func(img_vis),    # Ảnh với bounding boxes và labels (để debug)
            'characters': characters
        }
        
        return result

