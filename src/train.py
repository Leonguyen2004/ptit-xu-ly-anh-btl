import numpy as np
import os
import time
import matplotlib.pyplot as plt
from input import load_custom_data, unzip_dataset, split_train_val
from model import CNNModel

# ===== CẤU HÌNH =====
# Đường dẫn tới file RAR trên Google Drive
RAR_PATH = "/content/drive/MyDrive/Colab/dataset/CNN_letter_Dataset.rar"

# Thư mục để giải nén ra trên Colab
UNZIP_DIR = "/content/dataset"

# Nơi lưu model sau khi train
MODEL_SAVE_DIR = "/content/drive/MyDrive/Colab/model_output"
MODEL_NAME = "my_cnn_model.npz"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)

# Hyperparameters
EPOCHS = 15            # Số vòng lặp training
LEARNING_RATE = 0.005  # Tốc độ học
VAL_SPLIT = 0.2        # Dành 20% dữ liệu để kiểm tra (Validation)

def find_data_root(start_dir):
    """
    Hàm thông minh tự tìm thư mục chứa các class (0, 1, A, B...)
    để tránh lỗi sai đường dẫn sau khi giải nén.
    """
    # Kiểm tra ngay thư mục gốc
    subdirs = [d for d in os.listdir(start_dir) if os.path.isdir(os.path.join(start_dir, d))]
    # Nếu thấy folder tên là "0" hoặc "A" ngay đây thì đúng là nó
    if "0" in subdirs or "A" in subdirs:
        return start_dir

    # Nếu không, thử đi sâu vào 1 cấp (trường hợp giải nén ra folder cha)
    for d in subdirs:
        next_level = os.path.join(start_dir, d)
        sub_subdirs = [s for s in os.listdir(next_level) if os.path.isdir(os.path.join(next_level, s))]
        if "0" in sub_subdirs or "A" in sub_subdirs:
            return next_level

    return start_dir # Fallback

def plot_history(history, save_dir):
    """Vẽ biểu đồ và lưu lại"""
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Val Loss', color='red')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc', color='blue')
    plt.plot(history['val_acc'], label='Val Acc', color='red')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    chart_path = os.path.join(save_dir, 'training_result_chart.png')
    plt.savefig(chart_path)
    print(f"\n[CHART] Đã lưu biểu đồ đánh giá tại: {chart_path}")
    # plt.show()

def train_step(model, img, lbl, lr):
    # Forward
    probs, _ = model.forward(img, training=True)

    # Tính Loss
    loss = -np.log(probs[lbl] + 1e-9)
    acc = 1 if np.argmax(probs) == lbl else 0

    # Backward
    grad_logits = probs.copy()
    grad_logits[lbl] -= 1

    # Truyền ngược (Backpropagation)
    d = model.dense2.backprop(grad_logits, lr)
    d = model.dropout1.backprop(d)
    d = model.relu2.backprop(d)
    d = model.dense1.backprop(d, lr)
    d = model.flatten_layer.backprop(d)
    d = model.pool1.backprop(d)
    d = model.relu1.backprop(d)
    model.conv1.backprop(d, lr)

    return loss, acc

def evaluate(model, images, labels):
    """Đánh giá model trên tập Validation"""
    loss_sum, correct = 0, 0
    for img, lbl in zip(images, labels):
        probs, _ = model.forward(img, training=False)
        loss_sum += -np.log(probs[lbl] + 1e-9)
        if np.argmax(probs) == lbl: correct += 1
    return loss_sum/len(images), (correct/len(images))*100

def main():
    print("=== BẮT ĐẦU QUÁ TRÌNH TRAINING ===")

    # 1. Giải nén Dataset
    if not unzip_dataset(RAR_PATH, UNZIP_DIR):
        print(f"[LỖI] Không thể giải nén file: {RAR_PATH}")
        print("Hãy kiểm tra kỹ đường dẫn file RAR trên Google Drive của bạn.")
        return

    # 2. Xác định thư mục chứa data
    print("[INFO] Đang tìm thư mục dữ liệu...")
    data_dir = find_data_root(UNZIP_DIR)
    print(f"[INFO] Đã tìm thấy dữ liệu tại: {data_dir}")

    # 3. Load toàn bộ dữ liệu
    print(f"[INFO] Đang load ảnh và nhãn từ {data_dir}...")
    # Lưu ý: dataset của bạn không chia train/test nên ta load hết 1 cục
    images, labels, num_classes, idx_to_class, _ = load_custom_data(data_dir)

    if len(images) == 0:
        print(f"[LỖI] Không tìm thấy ảnh nào! Hãy kiểm tra lại file nén.")
        return

    # 4. Chia Train / Validation tự động (80% train, 20% val)
    print(f"[INFO] Đang chia dữ liệu (Validation Split = {VAL_SPLIT})...")
    train_X, train_y, val_X, val_y = split_train_val(images, labels, val_split=VAL_SPLIT)

    print("-" * 50)
    print(f"Tổng số Class: {num_classes}")
    print(f"Dữ liệu Train: {len(train_X)} ảnh")
    print(f"Dữ liệu Valid: {len(val_X)} ảnh")
    print(f"Kích thước ảnh input: {train_X[0].shape}")
    print("-" * 50)

    # 5. Khởi tạo Model
    model = CNNModel(input_shape=train_X[0].shape, num_classes=num_classes)

    # Lưu lịch sử để vẽ biểu đồ
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # 6. Training Loop
    for epoch in range(EPOCHS):
        # Xáo trộn dữ liệu mỗi epoch
        perm = np.random.permutation(len(train_X))
        train_X, train_y = train_X[perm], train_y[perm]

        epoch_loss, epoch_acc = 0, 0
        start_time = time.time()

        # Train từng ảnh (SGD)
        for i, (img, lbl) in enumerate(zip(train_X, train_y)):
            l, a = train_step(model, img, lbl, LEARNING_RATE)
            epoch_loss += l
            epoch_acc += a

            # In tiến độ mỗi 200 ảnh
            if i % 200 == 0:
                percent = (i / len(train_X)) * 100
                print(f"\r  Epoch {epoch+1} Running... {percent:.1f}%", end="")

        # Tổng kết Epoch
        train_avg_loss = epoch_loss / len(train_X)
        train_avg_acc = (epoch_acc / len(train_X)) * 100

        # Đánh giá trên tập Validation
        val_loss, val_acc = evaluate(model, val_X, val_y)

        # Lưu history
        history['train_loss'].append(train_avg_loss)
        history['train_acc'].append(train_avg_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"\rEpoch {epoch+1:02d}/{EPOCHS} | "
              f"Train Loss: {train_avg_loss:.4f} Acc: {train_avg_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
              f"Time: {time.time()-start_time:.1f}s")

    # 7. Lưu Model và Biểu đồ
    print("\n[INFO] Đang lưu kết quả...")
    model.save_model(MODEL_SAVE_PATH, idx_to_class)
    plot_history(history, MODEL_SAVE_DIR)
    print("\n=== HOÀN TẤT ===")

if __name__ == '__main__':
    main()