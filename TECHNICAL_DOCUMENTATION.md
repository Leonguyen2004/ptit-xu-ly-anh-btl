# Tài Liệu Kỹ Thuật - License Plate Recognition System

## Mục Lục
1. [Tổng Quan Hệ Thống](#tổng-quan-hệ-thống)
2. [Kiến Trúc Hệ Thống](#kiến-trúc-hệ-thống)
3. [Chi Tiết Code Backend](#chi-tiết-code-backend)
4. [Chi Tiết Code Frontend](#chi-tiết-code-frontend)
5. [Chi Tiết LeoPipeline](#chi-tiết-leopipeline)
6. [Luồng Dữ Liệu Hoàn Chỉnh](#luồng-dữ-liệu-hoàn-chỉnh)
7. [Cấu Trúc Dữ Liệu](#cấu-trúc-dữ-liệu)

---

## Tổng Quan Hệ Thống

Hệ thống License Plate Recognition (LPR) là một ứng dụng web cho phép:
- Upload và nhận diện biển số xe
- Xử lý ảnh với các thuật toán computer vision
- Chỉnh sửa kết quả nhận diện
- Lưu trữ dữ liệu ký tự để training

**Tech Stack:**
- **Backend**: Flask (Python)
- **Frontend**: Vanilla JavaScript, HTML, CSS
- **AI Models**: YOLO (detection), CNN (classification)
- **Image Processing**: OpenCV, NumPy

---

## Kiến Trúc Hệ Thống

```
┌─────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                        │
│                     (HTML + CSS + JS)                        │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ HTTP Requests (FormData/JSON)
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                      FLASK SERVER                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   /api/      │  │   /api/      │  │   /api/      │      │
│  │  pipeline    │  │process_image │  │save_characters│     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │              │
│         ▼                  ▼                  ▼              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ LeoPipeline  │  │Image Process │  │File System   │      │
│  └──────┬───────┘  └──────────────┘  └──────────────┘      │
└─────────┼──────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                       AI MODELS                              │
│  ┌──────────────────┐        ┌──────────────────┐          │
│  │  YOLO Detector   │   →    │  CNN Classifier  │          │
│  │(character_detector│        │(character_       │          │
│  │     .pt)         │        │ classifier.v2.npz)│          │
│  └──────────────────┘        └──────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

---

## Chi Tiết Code Backend

### File: `app.py`

#### 1. **Khởi Tạo và Cấu Hình**

```python
app = Flask(__name__, static_folder='static', template_folder='static')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pipeline = LPRPipeline(BASE_DIR)
leo_pipeline = LeoPipeline(BASE_DIR)
```

**Giải thích:**
- `Flask(__name__)`: Tạo Flask application instance
- `static_folder='static'`: Chỉ định thư mục chứa CSS, JS, images
- `BASE_DIR`: Đường dẫn tuyệt đối đến thư mục gốc của project
- `leo_pipeline`: Instance của LeoPipeline để xử lý nhận diện ký tự

---

#### 2. **Helper Functions**

##### `decode_image(file)`
```python
def decode_image(file):
    npimg = np.frombuffer(file.read(), np.uint8)
    return cv2.imdecode(npimg, cv2.IMREAD_COLOR)
```

**Chức năng:** Chuyển đổi file upload thành OpenCV image array

**Luồng xử lý:**
1. `file.read()`: Đọc binary data từ file upload
2. `np.frombuffer()`: Chuyển binary → NumPy array (uint8)
3. `cv2.imdecode()`: Decode array → BGR image (OpenCV format)

**Input:** Flask FileStorage object  
**Output:** NumPy array shape `(height, width, 3)` - BGR format

---

##### `encode_image(image)`
```python
def encode_image(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')
```

**Chức năng:** Chuyển đổi OpenCV image thành base64 string để gửi về frontend

**Luồng xử lý:**
1. `cv2.imencode('.jpg', image)`: Encode image → JPEG format (bytes)
2. `base64.b64encode()`: Encode bytes → base64
3. `.decode('utf-8')`: Chuyển bytes → string

**Input:** NumPy array (OpenCV image)  
**Output:** Base64 string (có thể embed vào HTML `<img>`)

---

#### 3. **API Endpoints**

##### `GET /`
```python
@app.route('/')
def index():
    return render_template('index.html')
```

**Chức năng:** Serve trang chủ của ứng dụng

---

##### `POST /api/pipeline`
```python
@app.route('/api/pipeline', methods=['POST'])
def run_pipeline():
    img = decode_image(request.files['image'])
    result = leo_pipeline.process(img, encode_image)
    return jsonify(result)
```

**Chức năng:** Endpoint chính để nhận diện biển số xe

**Luồng xử lý:**
1. Nhận file ảnh từ FormData
2. Decode thành OpenCV image
3. Gọi `leo_pipeline.process()` để:
   - Detect ký tự (YOLO)
   - Classify ký tự (CNN)
   - Sắp xếp ký tự
4. Return JSON response với:
   - `plate_text`: Chuỗi biển số
   - `plate_conf`: Độ tin cậy
   - `result_image`: Ảnh với bounding boxes (base64)
   - `characters`: Array thông tin từng ký tự

**Request:**
```
POST /api/pipeline
Content-Type: multipart/form-data

image: <file>
mode: "accurate" | "fast"
```

**Response:**
```json
{
  "plate_text": "30A12345",
  "plate_conf": 0.95,
  "result_image": "data:image/jpeg;base64,...",
  "visualization": "data:image/jpeg;base64,...",
  "characters": [
    {
      "label": "3",
      "conf": 0.98,
      "box": [10, 20, 30, 50],
      "crop": "base64_image_data"
    },
    ...
  ]
}
```

---

##### `POST /api/process_image`
```python
@app.route('/api/process_image', methods=['POST'])
def process_image():
    operation = request.form.get('operation')
    params = json.loads(request.form.get('params', '{}'))
    img = decode_image(request.files['image'])
    
    if operation == 'grayscale':
        processed = to_grayscale(img)
    elif operation == 'brightness_contrast':
        processed = adjust_brightness_contrast(img, 
            brightness=int(params.get('brightness', 0)),
            contrast=int(params.get('contrast', 0)))
    # ... các operations khác
    
    return jsonify({'image': encode_image(processed)})
```

**Chức năng:** Xử lý ảnh với các thuật toán computer vision

**Các operations hỗ trợ:**
- `grayscale`: Chuyển sang ảnh xám
- `brightness_contrast`: Điều chỉnh độ sáng/tương phản
- `gaussian_blur`: Làm mờ Gaussian
- `median_blur`: Làm mờ Median
- `canny`: Phát hiện cạnh Canny
- `threshold_otsu`: Ngưỡng hóa Otsu
- `threshold_adaptive`: Ngưỡng hóa thích ứng

**Request:**
```
POST /api/process_image
Content-Type: multipart/form-data

image: <file>
operation: "grayscale"
params: "{\"brightness\": 50, \"contrast\": 30}"
```

**Response:**
```json
{
  "image": "base64_encoded_processed_image"
}
```

---

##### `POST /api/save_characters`
```python
@app.route('/api/save_characters', methods=['POST'])
def save_characters():
    data = request.get_json()
    characters = data['characters']
    
    images_dir = os.path.join(BASE_DIR, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    saved_count = 0
    timestamp = int(os.path.getmtime(__file__) * 1000)
    
    for idx, char_data in enumerate(characters):
        label = char_data.get('label', '').strip()
        crop_base64 = char_data.get('crop', '')
        
        # Tạo thư mục cho label
        label_dir = os.path.join(images_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        
        # Decode và lưu ảnh
        img_data = base64.b64decode(crop_base64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        filename = f"{timestamp}_{idx}.jpg"
        filepath = os.path.join(label_dir, filename)
        cv2.imwrite(filepath, img)
        saved_count += 1
    
    return jsonify({
        'success': True,
        'saved_count': saved_count,
        'message': f'Saved {saved_count} characters successfully'
    })
```

**Chức năng:** Lưu ảnh ký tự đã chỉnh sửa vào local storage

**Luồng xử lý:**
1. Nhận JSON data với array characters
2. Tạo thư mục `images/` nếu chưa có
3. Với mỗi ký tự:
   - Tạo thư mục `images/{label}/`
   - Decode base64 → OpenCV image
   - Lưu file với tên `{timestamp}_{index}.jpg`
4. Return số lượng ảnh đã lưu

**Cấu trúc thư mục output:**
```
images/
├── A/
│   ├── 1732419600000_0.jpg
│   └── 1732419600000_5.jpg
├── B/
│   └── 1732419600000_1.jpg
├── 0/
│   └── 1732419600000_2.jpg
└── ...
```

**Request:**
```json
{
  "characters": [
    {
      "label": "A",
      "crop": "base64_image_data"
    },
    ...
  ]
}
```

**Response:**
```json
{
  "success": true,
  "saved_count": 8,
  "message": "Saved 8 characters successfully"
}
```

---

## Chi Tiết Code Frontend

### File: `script.js`

#### 1. **State Management**

```javascript
let currentImage = null;      // File object của ảnh hiện tại
let currentMode = 'detect';   // Mode: 'detect' hoặc 'process'
```

**Giải thích:**
- `currentImage`: Lưu trữ file ảnh để gửi lên server
- `currentMode`: Xác định tab nào đang active

---

#### 2. **DOM Elements References**

```javascript
const fileInput = document.getElementById('file-input');
const originalImage = document.getElementById('original-image');
const resultImage = document.getElementById('result-image');
const resultsPanel = document.getElementById('results-panel');
// ... các elements khác
```

**Mục đích:** Cache DOM elements để tránh query lại nhiều lần

---

#### 3. **Event Listeners**

```javascript
fileInput.addEventListener('change', handleFileUpload);
runDetectionBtn.addEventListener('click', runDetection);
runProcessBtn.addEventListener('click', runProcessing);
procOperationSelect.addEventListener('change', updateDynamicParams);

// Save Characters button
document.addEventListener('click', (e) => {
    if (e.target.id === 'save-characters-btn' || 
        e.target.closest('#save-characters-btn')) {
        saveCharacters();
    }
});
```

**Giải thích:**
- Event delegation cho Save button (vì button được tạo động)
- `e.target.closest()`: Tìm parent element gần nhất match selector

---

#### 4. **Core Functions**

##### `handleFileUpload(e)`
```javascript
function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    currentImage = file;
    fileNameDisplay.textContent = file.name;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        originalImage.src = e.target.result;
        originalImage.classList.remove('hidden');
        placeholderText.classList.add('hidden');
        imageWrapperOriginal.classList.add('has-image');
        
        // Reset result
        resultImage.classList.add('hidden');
        resultsPanel.classList.add('hidden');
    };
    reader.readAsDataURL(file);
}
```

**Chức năng:** Xử lý khi user upload ảnh

**Luồng xử lý:**
1. Lấy file từ input event
2. Lưu vào `currentImage` state
3. Hiển thị tên file
4. Sử dụng FileReader để đọc file as Data URL
5. Set src của `<img>` element
6. Update UI: ẩn placeholder, hiện ảnh
7. Reset kết quả cũ

**FileReader.readAsDataURL():**
- Đọc file và convert thành base64 data URL
- Format: `data:image/jpeg;base64,/9j/4AAQSkZJRg...`
- Có thể set trực tiếp vào `img.src`

---

##### `runDetection()`
```javascript
async function runDetection() {
    if (!currentImage) {
        alert('Please upload an image first');
        return;
    }
    
    showLoading(true);
    resultsPanel.classList.add('hidden');
    
    const formData = new FormData();
    formData.append('image', currentImage);
    formData.append('mode', detectionModeSelect.value);
    
    try {
        const response = await fetch('/api/pipeline', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayDetectionResults(data);
        } else {
            alert('Error: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred during detection');
    } finally {
        showLoading(false);
    }
}
```

**Chức năng:** Gửi ảnh lên server để nhận diện

**Luồng xử lý:**
1. Validate: kiểm tra có ảnh chưa
2. Hiển thị loading spinner
3. Tạo FormData với file ảnh và mode
4. Gửi POST request đến `/api/pipeline`
5. Parse JSON response
6. Nếu thành công: hiển thị kết quả
7. Nếu lỗi: hiển thị error message
8. Finally: ẩn loading spinner

**FormData:**
- API để tạo multipart/form-data request
- Cho phép upload file qua HTTP
- `formData.append(key, value)`: thêm field

---

##### `displayDetectionResults(data)`
```javascript
function displayDetectionResults(data) {
    // Hiển thị result image
    if (data.result_image) {
        resultImage.src = 'data:image/jpeg;base64,' + data.result_image;
        resultImage.classList.remove('hidden');
        imageWrapperResult.classList.add('has-image');
    }
    
    // Hiển thị plate text và confidence
    document.getElementById('plate-text').textContent = data.plate_text || '---';
    document.getElementById('plate-conf').textContent = 
        (data.plate_conf * 100).toFixed(1) + '%';
    
    // Render character cards với editable inputs
    const grid = document.getElementById('chars-grid');
    grid.innerHTML = '';
    
    if (data.characters && data.characters.length > 0) {
        data.characters.forEach((char, idx) => {
            const card = document.createElement('div');
            card.className = 'char-card';
            card.dataset.index = idx;
            card.innerHTML = `
                <img src="data:image/jpeg;base64,${char.crop}" alt="Character ${idx}">
                <input type="text" class="char-label-input" 
                       value="${char.label}" 
                       maxlength="3" 
                       data-original="${char.label}">
                <div class="char-conf">${(char.conf * 100).toFixed(0)}%</div>
            `;
            grid.appendChild(card);
        });
        
        // Lưu character data vào dataset để dùng khi save
        grid.dataset.characters = JSON.stringify(data.characters);
    }
    
    resultsPanel.classList.remove('hidden');
}
```

**Chức năng:** Hiển thị kết quả nhận diện lên UI

**Luồng xử lý:**
1. **Result Image:**
   - Set src với base64 image
   - Remove class 'hidden'

2. **Plate Info:**
   - Update text content với biển số
   - Format confidence thành %

3. **Character Cards:**
   - Clear grid hiện tại
   - Với mỗi character:
     - Tạo div.char-card
     - Set dataset.index
     - Render HTML với:
       - `<img>`: Ảnh ký tự crop
       - `<input>`: Editable label (KEY FEATURE!)
       - `<div>`: Confidence %
   - Append vào grid

4. **Store Data:**
   - Lưu toàn bộ characters data vào `grid.dataset.characters`
   - Dùng JSON.stringify để convert object → string
   - Sẽ dùng lại khi save

**Key Point:** Input field cho phép user chỉnh sửa label trước khi save!

---

##### `saveCharacters()`
```javascript
async function saveCharacters() {
    const grid = document.getElementById('chars-grid');
    const cards = grid.querySelectorAll('.char-card');
    
    if (cards.length === 0) {
        alert('No characters to save');
        return;
    }
    
    // Thu thập character data với edited labels
    const charactersData = JSON.parse(grid.dataset.characters || '[]');
    const characters = [];
    
    cards.forEach((card, idx) => {
        const input = card.querySelector('.char-label-input');
        const label = input.value.trim();
        
        if (label && charactersData[idx]) {
            characters.push({
                label: label,              // Label đã edit
                crop: charactersData[idx].crop  // Base64 image gốc
            });
        }
    });
    
    if (characters.length === 0) {
        alert('No valid characters to save');
        return;
    }
    
    // Hiển thị loading state
    const saveBtn = document.getElementById('save-characters-btn');
    const originalText = saveBtn.innerHTML;
    saveBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Saving...';
    saveBtn.disabled = true;
    
    try {
        const response = await fetch('/api/save_characters', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ characters })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            alert(`✓ ${data.message}`);
            // Animation: flash green
            cards.forEach(card => {
                card.classList.add('saved');
                setTimeout(() => card.classList.remove('saved'), 2000);
            });
        } else {
            alert('Error: ' + (data.error || 'Failed to save characters'));
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while saving characters');
    } finally {
        // Restore button state
        saveBtn.innerHTML = originalText;
        saveBtn.disabled = false;
    }
}
```

**Chức năng:** Thu thập labels đã chỉnh sửa và lưu ảnh về local

**Luồng xử lý:**

1. **Validation:**
   - Kiểm tra có character cards không

2. **Data Collection:**
   - Parse stored characters data từ dataset
   - Với mỗi card:
     - Lấy value từ input (label đã edit)
     - Lấy crop image từ stored data
     - Push vào array nếu valid

3. **UI Feedback:**
   - Disable button
   - Hiển thị spinner icon
   - Text: "Saving..."

4. **API Call:**
   - POST request đến `/api/save_characters`
   - Content-Type: application/json
   - Body: JSON với characters array

5. **Response Handling:**
   - Success: 
     - Alert success message
     - Add class 'saved' → trigger CSS animation
     - Remove class sau 2s
   - Error:
     - Alert error message

6. **Cleanup (finally):**
   - Restore button text
   - Enable button

**Key Points:**
- Kết hợp edited labels (từ input) với original images (từ dataset)
- Loading state để UX tốt hơn
- Animation feedback khi save thành công

---

## Chi Tiết LeoPipeline

### File: `tools/leo_pipeline/leo_pipeline.py`

#### Class: `LeoPipeline`

##### `__init__(base_dir)`
```python
def __init__(self, base_dir):
    self.base_dir = base_dir
    self.weights_dir = os.path.join(base_dir, 'src', 'weight')
    
    self.character_detector_path = os.path.join(
        self.weights_dir, 'character_detector.pt')
    self.character_classifier_path = os.path.join(
        self.weights_dir, 'character_classifier.v2.npz')
    
    self.yolo_model = None
    self.cnn_model = None
    self.idx_to_class = None
    
    self._load_models()
```

**Chức năng:** Khởi tạo pipeline và load models

**Luồng:**
1. Set đường dẫn đến thư mục weights
2. Xác định paths cho 2 models
3. Initialize model variables
4. Gọi `_load_models()` để load

---

##### `_load_models()`
```python
def _load_models(self):
    # Load YOLO model
    if not os.path.exists(self.character_detector_path):
        raise FileNotFoundError(f"Character detector not found")
    self.yolo_model = YOLO(self.character_detector_path)
    
    # Load CNN model
    if not os.path.exists(self.character_classifier_path):
        raise FileNotFoundError(f"Character classifier not found")
    
    self.cnn_model, self.idx_to_class = CNNModel.load_model(
        self.character_classifier_path)
    if self.cnn_model is None:
        raise RuntimeError(f"Failed to load CNN model")
```

**Chức năng:** Load 2 AI models

**Models:**
1. **YOLO (character_detector.pt):**
   - Object detection model
   - Detect vị trí các ký tự trong ảnh
   - Output: bounding boxes

2. **CNN (character_classifier.v2.npz):**
   - Classification model
   - Phân loại ký tự là gì (A-Z, 0-9)
   - Output: label + confidence

---

##### `calculate_iou(box1, box2)`
```python
def calculate_iou(self, box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Tính intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Tính union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union
```

**Chức năng:** Tính IoU (Intersection over Union) giữa 2 bounding boxes

**IoU Formula:**
```
IoU = Area of Intersection / Area of Union
```

**Ví dụ:**
```
Box1: [10, 10, 50, 50]  (40x40 = 1600 pixels)
Box2: [30, 30, 70, 70]  (40x40 = 1600 pixels)

Intersection: [30, 30, 50, 50] (20x20 = 400 pixels)
Union: 1600 + 1600 - 400 = 2800 pixels

IoU = 400 / 2800 = 0.143
```

**Mục đích:** Xác định 2 boxes có overlap nhiều không → loại bỏ duplicate

---

##### `remove_duplicate_detections(detections, iou_threshold=0.5)`
```python
def remove_duplicate_detections(self, detections, iou_threshold=0.5):
    if not detections or len(detections) <= 1:
        return detections
    
    # Sắp xếp theo confidence giảm dần
    sorted_detections = sorted(detections, 
                               key=lambda x: x['conf'], 
                               reverse=True)
    
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
            
            # Nếu IoU > threshold, loại bỏ box có conf thấp hơn
            if iou > iou_threshold:
                keep_flags[j] = False
    
    return filtered_detections
```

**Chức năng:** Loại bỏ các detections trùng lặp

**Thuật toán (Non-Maximum Suppression - NMS):**
1. Sort detections theo confidence (cao → thấp)
2. Với mỗi detection:
   - Nếu đã bị mark loại bỏ → skip
   - Thêm vào filtered list
   - So sánh với tất cả detections sau nó:
     - Tính IoU
     - Nếu IoU > threshold → mark detection sau để loại bỏ
3. Return filtered list

**Ví dụ:**
```
Input: [
  {box: [10,10,50,50], conf: 0.9},  # A
  {box: [12,12,52,52], conf: 0.85}, # B (overlap với A)
  {box: [100,100,140,140], conf: 0.95} # C (xa A,B)
]

IoU(A, B) = 0.7 > 0.5 → Loại B (conf thấp hơn)
IoU(A, C) = 0.0 < 0.5 → Giữ C

Output: [A, C]
```

---

##### `sort_characters_left_to_right_top_to_bottom(detections)`
```python
def sort_characters_left_to_right_top_to_bottom(self, detections):
    if not detections:
        return []
    
    # Tính trung bình y-coordinate
    avg_y = sum((det['box'][1] + det['box'][3]) / 2 
                for det in detections) / len(detections)
    
    # Phân loại thành hàng trên và hàng dưới
    top_row = []
    bottom_row = []
    
    for det in detections:
        y_center = (det['box'][1] + det['box'][3]) / 2
        if y_center < avg_y:
            top_row.append(det)
        else:
            bottom_row.append(det)
    
    # Sắp xếp mỗi hàng từ trái sang phải
    top_row.sort(key=lambda x: x['box'][0])
    bottom_row.sort(key=lambda x: x['box'][0])
    
    # Kết hợp: hàng trên trước, sau đó hàng dưới
    return top_row + bottom_row
```

**Chức năng:** Sắp xếp ký tự theo thứ tự đọc (trái→phải, trên→dưới)

**Thuật toán:**
1. Tính y_center trung bình của tất cả boxes
2. Phân loại:
   - y_center < avg_y → top_row
   - y_center >= avg_y → bottom_row
3. Sort mỗi row theo x-coordinate (trái→phải)
4. Concat: top_row + bottom_row

**Ví dụ (biển số 2 hàng):**
```
Input boxes:
  [3] [0] [A]     ← top row (y_center < avg_y)
  [1] [2] [3] [4] [5] ← bottom row (y_center >= avg_y)

avg_y = (y_top + y_bottom) / 2

Output order: [3, 0, A, 1, 2, 3, 4, 5]
→ Biển số: "30A12345"
```

---

##### `recognize_character_cnn(char_image)`
```python
def recognize_character_cnn(self, char_image):
    if self.cnn_model is None:
        return None, 0.0
    
    try:
        # Preprocess: BGR → Grayscale
        if len(char_image.shape) == 3:
            gray = cv2.cvtColor(char_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = char_image
        
        # Resize về 28x28
        img_pil = PILImage.fromarray(gray)
        img_resized = img_pil.resize((28, 28), PILImage.LANCZOS)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        
        # Forward pass qua CNN
        probs, _ = self.cnn_model.forward(img_array, training=False)
        pred_idx = np.argmax(probs)
        pred_label = self.idx_to_class[pred_idx]
        confidence = float(probs[pred_idx])
        
        return pred_label, confidence
    except Exception as e:
        print(f"Error in CNN recognition: {e}")
        return None, 0.0
```

**Chức năng:** Phân loại ký tự bằng CNN model

**Preprocessing Pipeline:**
1. **Color Conversion:**
   - BGR (3 channels) → Grayscale (1 channel)
   - CNN model chỉ nhận grayscale

2. **Resize:**
   - Resize về 28x28 pixels (input size của CNN)
   - LANCZOS: high-quality resampling

3. **Normalization:**
   - Chia cho 255.0 → scale [0, 1]
   - Neural networks hoạt động tốt hơn với normalized input

4. **Inference:**
   - Forward pass qua CNN
   - Output: probability distribution (36 classes: A-Z, 0-9)
   - `np.argmax()`: lấy class có prob cao nhất
   - Map index → label qua `idx_to_class`

**Input:** Cropped character image (any size, BGR/Gray)  
**Output:** (label, confidence) - ví dụ: ("A", 0.98)

---

##### `process(image, encode_image_func)` - MAIN PIPELINE
```python
def process(self, image, encode_image_func):
    # 1. YOLO Detection
    results = self.yolo_model(image, conf=0.25, iou=0.5, verbose=False)
    
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id == 31:  # Skip license plate class
                continue
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = float(box.conf[0].cpu().numpy())
            
            # Clamp coordinates
            h, w = image.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # Crop character
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
    
    # 2. Remove duplicates
    filtered_detections = self.remove_duplicate_detections(
        detections, iou_threshold=0.5)
    
    if not filtered_detections:
        return {'error': 'No valid characters after filtering'}
    
    # 3. Sort characters
    sorted_detections = self.sort_characters_left_to_right_top_to_bottom(
        filtered_detections)
    
    # 4. Classify each character
    characters = []
    plate_text = ""
    
    for det in sorted_detections:
        label, cnn_conf = self.recognize_character_cnn(det['crop'])
        
        if label is None:
            label = "?"
            cnn_conf = 0.0
        
        characters.append({
            'label': label,
            'conf': float(cnn_conf),
            'box': det['box'],
            'crop': encode_image_func(det['crop'])
        })
        
        plate_text += label
    
    # 5. Create result image (with bounding boxes)
    img_result = image.copy()
    for char in characters:
        x1, y1, x2, y2 = char['box']
        cv2.rectangle(img_result, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 6. Create visualization (with boxes + labels)
    img_vis = image.copy()
    for char in characters:
        x1, y1, x2, y2 = char['box']
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_vis, char['label'], (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Format response
    result = {
        'plate_text': plate_text,
        'plate_conf': float(np.mean([char['conf'] for char in characters])),
        'result_image': encode_image_func(img_result),
        'visualization': encode_image_func(img_vis),
        'characters': characters
    }
    
    return result
```

**Chức năng:** Pipeline chính để nhận diện biển số

**6 Bước Xử Lý:**

**Bước 1: YOLO Detection**
- Run YOLO model trên ảnh
- Parameters:
  - `conf=0.25`: confidence threshold
  - `iou=0.5`: IoU threshold cho NMS
- Với mỗi detection:
  - Extract bounding box coordinates
  - Extract confidence score
  - Clamp coordinates trong phạm vi ảnh
  - Crop ký tự từ ảnh gốc
  - Lưu vào detections list

**Bước 2: Remove Duplicates**
- Gọi `remove_duplicate_detections()`
- Loại bỏ các boxes overlap quá nhiều
- Giữ lại box có confidence cao hơn

**Bước 3: Sort Characters**
- Gọi `sort_characters_left_to_right_top_to_bottom()`
- Sắp xếp theo thứ tự đọc

**Bước 4: Classify Characters**
- Với mỗi detection đã sort:
  - Gọi `recognize_character_cnn()` để classify
  - Encode crop image thành base64
  - Append vào characters array
  - Concat label vào plate_text

**Bước 5: Create Result Image**
- Copy ảnh gốc
- Vẽ bounding boxes (green, thickness 2)
- Không vẽ labels (clean visualization)

**Bước 6: Create Visualization Image**
- Copy ảnh gốc
- Vẽ bounding boxes
- Vẽ labels trên mỗi box (để debug)

**Output Structure:**
```json
{
  "plate_text": "30A12345",
  "plate_conf": 0.95,
  "result_image": "base64...",
  "visualization": "base64...",
  "characters": [
    {
      "label": "3",
      "conf": 0.98,
      "box": [10, 20, 30, 50],
      "crop": "base64..."
    },
    ...
  ]
}
```

---

## Luồng Dữ Liệu Hoàn Chỉnh

### Flow 1: Upload → Detection → Display

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. USER UPLOADS IMAGE                                           │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. FRONTEND (handleFileUpload)                                  │
│    - Get file from input                                        │
│    - Store in currentImage state                                │
│    - FileReader.readAsDataURL()                                 │
│    - Display preview in <img>                                   │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. USER CLICKS "Run Detection"                                  │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. FRONTEND (runDetection)                                      │
│    - Create FormData                                            │
│    - Append image file                                          │
│    - POST /api/pipeline                                         │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. BACKEND (run_pipeline)                                       │
│    - Receive FormData                                           │
│    - decode_image() → OpenCV array                              │
│    - Call leo_pipeline.process()                                │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. LEOPIPELINE (process)                                        │
│    Step 1: YOLO Detection                                       │
│      - Detect character bounding boxes                          │
│      - Crop each character                                      │
│    Step 2: Remove Duplicates                                    │
│      - Calculate IoU between boxes                              │
│      - Filter overlapping detections                            │
│    Step 3: Sort Characters                                      │
│      - Separate top/bottom rows                                 │
│      - Sort left to right                                       │
│    Step 4: Classify Characters                                  │
│      - Preprocess: resize 28x28, normalize                      │
│      - CNN forward pass                                         │
│      - Get label + confidence                                   │
│    Step 5: Create Result Image                                  │
│      - Draw bounding boxes                                      │
│    Step 6: Encode & Return                                      │
│      - encode_image() → base64                                  │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. BACKEND (run_pipeline)                                       │
│    - Receive result dict from pipeline                          │
│    - jsonify(result)                                            │
│    - Return HTTP 200 + JSON                                     │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 8. FRONTEND (runDetection)                                      │
│    - Receive JSON response                                      │
│    - Call displayDetectionResults(data)                         │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 9. FRONTEND (displayDetectionResults)                           │
│    - Display result_image                                       │
│    - Display plate_text + confidence                            │
│    - Render character cards:                                    │
│      * <img> with crop                                          │
│      * <input> with label (EDITABLE!)                           │
│      * <div> with confidence                                    │
│    - Store characters data in grid.dataset                      │
│    - Show results panel                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

### Flow 2: Edit Labels → Save

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. USER EDITS LABELS                                            │
│    - Click on input fields                                      │
│    - Modify character labels                                    │
│    - Example: "8" → "B", "0" → "O"                              │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. USER CLICKS "Save Characters"                                │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. FRONTEND (saveCharacters)                                    │
│    - Query all .char-card elements                              │
│    - Parse stored characters data from dataset                  │
│    - For each card:                                             │
│      * Get edited label from input.value                        │
│      * Get original crop from stored data                       │
│      * Push to characters array                                 │
│    - Show loading state on button                               │
│    - POST /api/save_characters with JSON                        │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. BACKEND (save_characters)                                    │
│    - Receive JSON with characters array                         │
│    - Create images/ directory                                   │
│    - Generate timestamp                                         │
│    - For each character:                                        │
│      * Get label (edited)                                       │
│      * Create images/{label}/ directory                         │
│      * Decode base64 → OpenCV image                             │
│      * Save as {timestamp}_{idx}.jpg                            │
│      * Increment saved_count                                    │
│    - Return success response                                    │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. FRONTEND (saveCharacters)                                    │
│    - Receive success response                                   │
│    - Alert success message                                      │
│    - Add 'saved' class to cards → CSS animation                 │
│    - Remove 'saved' class after 2s                              │
│    - Restore button state                                       │
└─────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. FILE SYSTEM                                                  │
│    images/                                                      │
│    ├── A/                                                       │
│    │   └── 1732419600000_0.jpg                                 │
│    ├── B/                                                       │
│    │   └── 1732419600000_1.jpg                                 │
│    └── ...                                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Cấu Trúc Dữ Liệu

### 1. Detection Object (trong LeoPipeline)
```python
{
    'box': [x1, y1, x2, y2],  # Bounding box coordinates
    'conf': 0.95,              # Detection confidence
    'crop': np.array(...)      # Cropped character image
}
```

### 2. Character Object (API Response)
```python
{
    'label': 'A',              # Character label
    'conf': 0.98,              # Classification confidence
    'box': [10, 20, 30, 50],   # Bounding box
    'crop': 'base64_string'    # Base64 encoded crop image
}
```

### 3. Pipeline Result (API Response)
```python
{
    'plate_text': '30A12345',           # Full plate text
    'plate_conf': 0.95,                 # Average confidence
    'result_image': 'base64_string',    # Image with boxes
    'visualization': 'base64_string',   # Image with boxes + labels
    'characters': [...]                 # Array of Character objects
}
```

### 4. Save Request (Frontend → Backend)
```json
{
  "characters": [
    {
      "label": "A",
      "crop": "base64_image_data"
    },
    ...
  ]
}
```

### 5. Save Response (Backend → Frontend)
```json
{
  "success": true,
  "saved_count": 8,
  "message": "Saved 8 characters successfully"
}
```

---

## Tổng Kết

Hệ thống hoạt động theo pipeline:
1. **Upload**: User upload ảnh → Frontend preview
2. **Detection**: YOLO detect ký tự → Crop từng ký tự
3. **Classification**: CNN classify từng ký tự → Nhận label
4. **Sorting**: Sắp xếp ký tự theo thứ tự đọc
5. **Display**: Hiển thị kết quả với editable labels
6. **Edit**: User chỉnh sửa labels nếu cần
7. **Save**: Lưu ảnh ký tự vào local storage theo label

**Key Features:**
- Real-time character recognition
- Editable labels for correction
- Organized storage for training data
- Modern, responsive UI
- Error handling và validation
