from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image, ImageEnhance
import io
import base64
import sqlite3
import csv
import datetime
import os
from utils import LicensePlateRecognizer
import cv2
import numpy as np

app = Flask(__name__)

# Database Setup
DB_NAME = "parking.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # History table - Simplified for just logging processed images
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  filename TEXT,
                  plate_text TEXT,
                  timestamp DATETIME,
                  image_path TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Initialize Recognizer
try:
    recognizer = LicensePlateRecognizer()
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    recognizer = None

@app.route('/')
def index():
    return render_template('index.html')

def preprocess_image(image_pil, brightness=1.0, contrast=1.0, rotation=0):
    # Rotation
    if rotation != 0:
        image_pil = image_pil.rotate(-rotation, expand=True)
    
    # Brightness
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(image_pil)
        image_pil = enhancer.enhance(brightness)
        
    # Contrast
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(image_pil)
        image_pil = enhancer.enhance(contrast)
        
    return image_pil

@app.route('/api/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Load and Preprocess
        image = Image.open(file.stream)
        
        # Get parameters
        yolo_conf = float(request.form.get('yolo_conf', 0.25))
        brightness = float(request.form.get('brightness', 1.0))
        contrast = float(request.form.get('contrast', 1.0))
        rotation = int(request.form.get('rotation', 0))
        
        # Apply preprocessing
        processed_image = preprocess_image(image, brightness, contrast, rotation)
        
        # Recognition
        result = recognizer.process_image(processed_image, yolo_conf=yolo_conf)
        
        if isinstance(result, tuple): # Error case
             return jsonify({'error': result[1]}), 500
        
        plate_text = result['text']
        
        # Save to History
        timestamp = datetime.datetime.now()
        
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("INSERT INTO history (filename, plate_text, timestamp) VALUES (?, ?, ?)",
                  (file.filename, plate_text, timestamp.isoformat()))
        conn.commit()
        conn.close()
             
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
        
    file = request.files['image']
    try:
        image = Image.open(file.stream)
        analysis_results = recognizer.analyze_image(image)
        return jsonify(analysis_results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT * FROM history ORDER BY timestamp DESC LIMIT 50")
    rows = c.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        history.append({
            'id': row[0],
            'filename': row[1],
            'plate_text': row[2],
            'timestamp': row[3]
        })
    return jsonify(history)

@app.route('/api/export', methods=['GET'])
def export_history():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT * FROM history ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    
    # Create CSV in memory
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['ID', 'Filename', 'Plate Text', 'Timestamp', 'Image Path'])
    cw.writerows(rows)
    
    output = io.BytesIO()
    output.write(si.getvalue().encode('utf-8-sig'))
    output.seek(0)
    
    return send_file(output, mimetype='text/csv', as_attachment=True, download_name='recognition_history.csv')

@app.route('/api/batch', methods=['POST'])
def batch_process():
    if 'images' not in request.files:
        return jsonify({'error': 'No images uploaded'}), 400
    
    files = request.files.getlist('images')
    results = []
    
    for file in files:
        try:
            image = Image.open(file.stream)
            res = recognizer.process_image(image)
            
            if not isinstance(res, tuple):
                results.append({
                    'filename': file.filename,
                    'text': res['text'],
                    'status': 'success'
                })
            else:
                results.append({
                    'filename': file.filename,
                    'error': res[1],
                    'status': 'error'
                })
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e),
                'status': 'error'
            })
            
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
