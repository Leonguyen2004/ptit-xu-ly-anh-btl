import os
import cv2
import numpy as np
import base64
import json
from flask import Flask, render_template, request, jsonify, send_file
from tools.pipeline.pipeline import LPRPipeline
from tools.leo_pipeline.leo_pipeline import LeoPipeline
from tools.image_processing.image_processing import (
    to_grayscale, gaussian_blur, median_blur, 
    canny_edge_detection, threshold_otsu, threshold_adaptive,
    adjust_brightness_contrast
)

app = Flask(__name__, static_folder='static', template_folder='static')

# Initialize Pipelines
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pipeline = LPRPipeline(BASE_DIR)
leo_pipeline = LeoPipeline(BASE_DIR)

def decode_image(file):
    npimg = np.frombuffer(file.read(), np.uint8)
    return cv2.imdecode(npimg, cv2.IMREAD_COLOR)

def encode_image(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    img = decode_image(request.files['image'])
    detections = pipeline.lp_detector.detect(img)
    
    # Draw boxes for visualization
    img_vis = img.copy()
    for det in detections:
        x1, y1, x2, y2 = det['box']
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
    return jsonify({
        'detections': detections,
        'image': encode_image(img_vis)
    })

@app.route('/api/pipeline', methods=['POST'])
def run_pipeline():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    img = decode_image(request.files['image'])
    
    try:
        # Xử lý ảnh bằng LeoPipeline
        result = leo_pipeline.process(img, encode_image)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
        
    operation = request.form.get('operation')
    params = json.loads(request.form.get('params', '{}'))
    
    img = decode_image(request.files['image'])
    processed = img
    
    try:
        if operation == 'grayscale':
            processed = to_grayscale(img)
        elif operation == 'brightness_contrast':
            processed = adjust_brightness_contrast(img, 
                                                 brightness=int(params.get('brightness', 0)),
                                                 contrast=int(params.get('contrast', 0)))
        elif operation == 'gaussian_blur':
            processed = gaussian_blur(img, kernel_size=int(params.get('kernel_size', 5)))
        elif operation == 'median_blur':
            processed = median_blur(img, kernel_size=int(params.get('kernel_size', 5)))
        elif operation == 'canny':
            processed = canny_edge_detection(img, 
                                           threshold1=int(params.get('threshold1', 100)),
                                           threshold2=int(params.get('threshold2', 200)))
        elif operation == 'threshold_otsu':
            processed = threshold_otsu(img)
        elif operation == 'threshold_adaptive':
            processed = threshold_adaptive(img, 
                                         block_size=int(params.get('block_size', 11)),
                                         C=int(params.get('C', 2)))
                                         
        return jsonify({
            'image': encode_image(processed)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/save_characters', methods=['POST'])
def save_characters():
    """
    Lưu các ký tự đã được chỉnh sửa label vào thư mục local.
    
    Request body (JSON):
    {
        "characters": [
            {
                "label": "A",
                "crop": "base64_encoded_image"
            },
            ...
        ]
    }
    
    Response:
    {
        "success": true,
        "saved_count": 8,
        "message": "Saved 8 characters successfully"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'characters' not in data:
            return jsonify({'error': 'No characters data provided'}), 400
        
        characters = data['characters']
        
        if not characters:
            return jsonify({'error': 'Characters list is empty'}), 400
        
        # Tạo thư mục images nếu chưa tồn tại
        images_dir = os.path.join(BASE_DIR, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        saved_count = 0
        timestamp = int(os.path.getmtime(__file__) * 1000)  # Use current timestamp
        
        for idx, char_data in enumerate(characters):
            label = char_data.get('label', '').strip()
            crop_base64 = char_data.get('crop', '')
            
            if not label or not crop_base64:
                continue
            
            # Tạo thư mục cho label nếu chưa tồn tại
            label_dir = os.path.join(images_dir, label)
            os.makedirs(label_dir, exist_ok=True)
            
            # Decode base64 image
            try:
                img_data = base64.b64decode(crop_base64)
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    continue
                
                # Tạo tên file unique: {timestamp}_{index}.jpg
                filename = f"{timestamp}_{idx}.jpg"
                filepath = os.path.join(label_dir, filename)
                
                # Lưu ảnh
                cv2.imwrite(filepath, img)
                saved_count += 1
                
            except Exception as e:
                print(f"Error saving character {idx}: {e}")
                continue
        
        return jsonify({
            'success': True,
            'saved_count': saved_count,
            'message': f'Saved {saved_count} characters successfully'
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
