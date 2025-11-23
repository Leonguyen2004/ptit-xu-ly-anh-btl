import os
from PIL import Image as PILImage
import numpy as np
import matplotlib.pyplot as plt
from model import CNNModel

# ===== CẤU HÌNH =====
# Đường dẫn đến file model (tương đối so với file script này)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "..", "weight", "my_cnn_model.v2.npz")

# Đường dẫn đến file ảnh cần dự đoán
IMAGE_PATH = os.path.join(CURRENT_DIR, "..", "images", "test", "0_2.png")

def predict_image(image_path):
    # 1. Load Model trước
    # Kiểm tra model path
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    model, idx_to_class = CNNModel.load_model(MODEL_PATH)
    if model is None: 
        print("Failed to load model.")
        return

    # 2. Kiểm tra file ảnh
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    try:
        # 3. Xử lý ảnh
        img_pil = PILImage.open(image_path).convert('L') # Chuyển sang đen trắng

        # Resize về đúng kích thước model đã học (28x28)
        img_resized = img_pil.resize((28, 28), PILImage.LANCZOS)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0 # Normalize

        # 4. Dự đoán
        probs, _ = model.forward(img_array, training=False)
        pred_idx = np.argmax(probs)
        pred_label = idx_to_class[pred_idx]
        confidence = probs[pred_idx] * 100

        # 5. Hiển thị kết quả
        print(f"\n-> File '{os.path.basename(image_path)}' được nhận diện là: {pred_label}")
        print(f"-> Độ tin cậy: {confidence:.1f}%")

        plt.figure(figsize=(3, 3))
        plt.imshow(img_array, cmap='gray')
        plt.title(f"Predict: {pred_label}\nConf: {confidence:.1f}%")
        plt.axis('off')
        plt.show()

    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    print("=== AI PREDICTION TOOL ===")
    print(f"Model path: {MODEL_PATH}")
    print(f"Image path: {IMAGE_PATH}")
    
    predict_image(IMAGE_PATH)