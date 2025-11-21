import numpy as np
import os
from PIL import Image as PILImage
import subprocess
from typing import Tuple, Dict

def unzip_dataset(rar_path, unzip_dir):
    """
    Giải nén file RAR dataset vào thư mục chỉ định.
    
    Args:
        rar_path (str): Đường dẫn đến file .rar
        unzip_dir (str): Thư mục đích để giải nén
        
    Returns:
        bool: True nếu giải nén thành công, False nếu thất bại
    """
    # Tạo thư mục đích nếu chưa tồn tại
    if not os.path.exists(unzip_dir):
        os.makedirs(unzip_dir)
        print(f"[INFO] Thư mục {unzip_dir} đã được tạo.")
    else:
        print(f"[INFO] Thư mục {unzip_dir} đã tồn tại.")

    # Kiểm tra file RAR có tồn tại không
    if not os.path.exists(rar_path):
        print(f"[ERROR] File {rar_path} không tồn tại.")
        print(f"[INFO] Đường dẫn tuyệt đối: {os.path.abspath(rar_path)}")
        return False

    print(f"[INFO] Đang giải nén {rar_path} vào {unzip_dir}...")
    
    try:
        # Sử dụng unrar với các options:
        # x: extract với full path
        # -o+: overwrite existing files
        process = subprocess.run(
            ['unrar', 'x', '-o+', rar_path, unzip_dir + os.sep],
            capture_output=True,
            text=True,
            check=True,
            timeout=300  # Timeout 5 phút để tránh treo
        )
        print("[SUCCESS] Giải nén thành công!")
        return True
        
    except FileNotFoundError:
        print("[ERROR] Lệnh 'unrar' không được tìm thấy.")
        print("[INFO] Hướng dẫn cài đặt:")
        print("  - Ubuntu/Debian: sudo apt-get install unrar")
        print("  - MacOS: brew install unrar")
        print("  - Windows: Download từ https://www.rarlab.com/")
        return False
        
    except subprocess.TimeoutExpired:
        print("[ERROR] Giải nén quá thời gian cho phép (5 phút).")
        return False
        
    except subprocess.CalledProcessError as e:
        print("[ERROR] Lỗi khi giải nén:")
        print(e.stderr if e.stderr else e.stdout)
        return False
        
    except Exception as e:
        print(f"[ERROR] Lỗi không xác định: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_custom_data(base_dir, target_size=(28, 28), max_images_per_class=None):
    """
    Load dataset từ cấu trúc thư mục (mỗi class là một thư mục con).
    
    Cấu trúc thư mục mong đợi:
    base_dir/
        class_1/
            image_1.jpg
            image_2.jpg
            ...
        class_2/
            image_1.jpg
            ...
    
    Args:
        base_dir (str): Đường dẫn đến thư mục chứa dataset
        target_size (tuple): Kích thước ảnh sau khi resize (height, width)
        max_images_per_class (int): Giới hạn số ảnh load cho mỗi class (None = load tất cả)
        
    Returns:
        tuple: (images, labels, num_classes, idx_to_class, class_to_idx)
            - images: numpy array shape (N, H, W) với giá trị [0, 1]
            - labels: numpy array shape (N,) với giá trị integer
            - num_classes: số lượng classes
            - idx_to_class: dict mapping index -> class name
            - class_to_idx: dict mapping class name -> index
    """
    images = []
    labels = []

    # Kiểm tra thư mục tồn tại
    if not os.path.exists(base_dir):
        print(f"[ERROR] Thư mục {base_dir} không tồn tại.")
        print(f"[INFO] Đường dẫn tuyệt đối: {os.path.abspath(base_dir)}")
        return np.array(images), np.array(labels), 0, {}, {}

    # Lấy danh sách các class (thư mục con)
    class_names = sorted([
        d for d in os.listdir(base_dir) 
        if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith('.')
    ])
    
    if not class_names:
        print(f"[ERROR] Không tìm thấy thư mục class nào trong {base_dir}.")
        return np.array(images), np.array(labels), 0, {}, {}

    # Tạo mapping giữa class name và index
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
    num_classes = len(class_names)

    print(f"\n[INFO] Tìm thấy {num_classes} classes: {class_names}")
    print(f"[INFO] Class mapping: {class_to_idx}")
    if max_images_per_class:
        print(f"[INFO] Giới hạn: {max_images_per_class} ảnh/class")

    # Supported image extensions
    SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')
    
    # Load images từ mỗi class
    total_images_loaded = 0
    failed_images = 0
    
    for class_name in class_names:
        class_dir = os.path.join(base_dir, class_name)
        class_image_count = 0
        
        # Lấy danh sách files trong class directory
        image_files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith(SUPPORTED_EXTENSIONS)
        ]
        
        # Giới hạn số lượng nếu được chỉ định
        if max_images_per_class:
            image_files = image_files[:max_images_per_class]
        
        # Load từng ảnh
        for image_name in image_files:
            image_path = os.path.join(class_dir, image_name)
            
            try:
                # Load ảnh và convert sang grayscale
                img = PILImage.open(image_path).convert('L')
                
                # Resize về target size
                img = img.resize(target_size, PILImage.LANCZOS)
                
                # Convert sang numpy array và normalize về [0, 1]
                img_array = np.array(img, dtype=np.float32) / 255.0
                
                # Kiểm tra shape hợp lệ
                if img_array.shape != target_size:
                    print(f"[WARNING] Ảnh {image_path} có shape không đúng: {img_array.shape}")
                    failed_images += 1
                    continue
                
                images.append(img_array)
                labels.append(class_to_idx[class_name])
                class_image_count += 1
                total_images_loaded += 1
                
            except Exception as e:
                print(f"[WARNING] Không thể load ảnh {image_path}: {e}")
                failed_images += 1
                continue
        
        print(f"  - Class '{class_name}': {class_image_count} ảnh")

    # Kiểm tra có ảnh được load không
    if not images:
        print(f"[ERROR] Không load được ảnh nào từ {base_dir}.")
        return np.array(images), np.array(labels), num_classes, idx_to_class, class_to_idx

    # Convert sang numpy arrays
    images_array = np.array(images, dtype=np.float32)
    labels_array = np.array(labels, dtype=np.int32)
    
    # In thông tin tổng kết
    print(f"\n[SUCCESS] Tổng cộng load được {total_images_loaded} ảnh")
    if failed_images > 0:
        print(f"[WARNING] {failed_images} ảnh bị lỗi khi load")
    print(f"[INFO] Images shape: {images_array.shape}")
    print(f"[INFO] Labels shape: {labels_array.shape}")
    print(f"[INFO] Phân bố classes:")
    
    # In phân bố số lượng ảnh theo class
    for idx in range(num_classes):
        count = np.sum(labels_array == idx)
        print(f"  - Class {idx} ({idx_to_class[idx]}): {count} ảnh")

    return images_array, labels_array, num_classes, idx_to_class, class_to_idx


def split_train_val(images, labels, val_split=0.2, random_seed=42):
    """
    Chia dataset thành training set và validation set.
    
    Args:
        images (np.array): Mảng ảnh
        labels (np.array): Mảng labels
        val_split (float): Tỉ lệ validation (0.0-1.0)
        random_seed (int): Random seed để reproducibility
        
    Returns:
        tuple: (train_images, train_labels, val_images, val_labels)
    """
    if not 0.0 < val_split < 1.0:
        raise ValueError(f"val_split phải trong khoảng (0, 1), nhận được: {val_split}")
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Tạo permutation
    num_samples = len(images)
    indices = np.random.permutation(num_samples)
    
    # Tính split point
    split_idx = int(num_samples * (1 - val_split))
    
    # Split indices
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # Split data
    train_images = images[train_indices]
    train_labels = labels[train_indices]
    val_images = images[val_indices]
    val_labels = labels[val_indices]
    
    print(f"\n[INFO] Dataset split:")
    print(f"  - Training: {len(train_images)} samples")
    print(f"  - Validation: {len(val_images)} samples")
    
    return train_images, train_labels, val_images, val_labels


def augment_image(image, augmentation_params=None):
    """
    Áp dụng data augmentation cho một ảnh.
    
    Args:
        image (np.array): Ảnh đầu vào (H x W)
        augmentation_params (dict): Các tham số augmentation
        
    Returns:
        np.array: Ảnh sau khi augment
    """
    if augmentation_params is None:
        augmentation_params = {
            'rotation': 10,      # Độ xoay tối đa (degrees)
            'shift': 2,          # Dịch chuyển tối đa (pixels)
            'noise_std': 0.01    # Độ lệch chuẩn của noise
        }
    
    augmented = image.copy()
    
    # Random rotation (nếu cần implement)
    # Có thể sử dụng scipy.ndimage.rotate
    
    # Random shift
    if augmentation_params.get('shift', 0) > 0:
        shift_x = np.random.randint(-augmentation_params['shift'], 
                                     augmentation_params['shift'] + 1)
        shift_y = np.random.randint(-augmentation_params['shift'], 
                                     augmentation_params['shift'] + 1)
        augmented = np.roll(augmented, shift_x, axis=0)
        augmented = np.roll(augmented, shift_y, axis=1)
    
    # Add random noise
    if augmentation_params.get('noise_std', 0) > 0:
        noise = np.random.normal(0, augmentation_params['noise_std'], augmented.shape)
        augmented = augmented + noise
        augmented = np.clip(augmented, 0, 1)  # Giữ trong khoảng [0, 1]
    
    return augmented