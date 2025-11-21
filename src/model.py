import numpy as np
import os
from layers import Conv, ReLU, MaxPool, Flatten, Dense, Dropout, Softmax

class CNNModel:
    """
    Convolutional Neural Network Model cho phân loại ảnh.
    
    Kiến trúc: Conv -> ReLU -> MaxPool -> Flatten -> Dense -> ReLU -> Dropout -> Dense -> Softmax
    
    Args:
        input_shape (tuple): Kích thước ảnh đầu vào (height, width)
        num_classes (int): Số lượng classes cần phân loại
        conv_filters (int): Số lượng filters trong Conv layer
        conv_kernel_size (int): Kích thước kernel của Conv layer
        pool_size (int): Kích thước pooling window
        pool_stride (int): Stride của pooling operation
        dense1_nodes (int): Số neurons trong Dense layer đầu tiên
        dropout_rate (float): Tỉ lệ dropout (0.0-1.0)
    """
    def __init__(self, input_shape=(28, 28), num_classes=10,
                 conv_filters=8, conv_kernel_size=3,
                 pool_size=2, pool_stride=2,
                 dense1_nodes=128, dropout_rate=0.5):

        # Lưu hyperparameters
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.dense1_nodes = dense1_nodes
        self.dropout_rate = dropout_rate

        # ===== Khởi tạo các layers =====
        # Layer 1: Convolutional layer
        self.conv1 = Conv(num_filters=self.conv_filters, filter_size=self.conv_kernel_size)
        # Layer 2: ReLU activation
        self.relu1 = ReLU()
        # Layer 3: Max pooling
        self.pool1 = MaxPool(pool_size=self.pool_size, stride=self.pool_stride)
        # Layer 4: Flatten để chuyển từ 3D sang 1D
        self.flatten_layer = Flatten()

        # ===== Tính toán kích thước sau các transformations =====
        # Sau Conv layer (valid convolution, no padding)
        conv_out_h = self.input_shape[0] - self.conv_kernel_size + 1
        conv_out_w = self.input_shape[1] - self.conv_kernel_size + 1
        
        # Sau MaxPool layer
        pool_out_h = (conv_out_h - self.pool_size) // self.pool_stride + 1
        pool_out_w = (conv_out_w - self.pool_size) // self.pool_stride + 1
        
        # Kích thước sau khi flatten
        self.flattened_size = pool_out_h * pool_out_w * self.conv_filters

        # ===== Khởi tạo fully connected layers =====
        # Layer 5: Dense layer đầu tiên
        self.dense1 = Dense(input_len=self.flattened_size, output_len=self.dense1_nodes)
        # Layer 6: ReLU activation
        self.relu2 = ReLU()
        # Layer 7: Dropout regularization
        self.dropout1 = Dropout(rate=self.dropout_rate)
        # Layer 8: Output layer
        self.dense2 = Dense(input_len=self.dense1_nodes, output_len=self.num_classes)
        # Layer 9: Softmax để tạo probability distribution
        self.softmax_activation = Softmax()

        print(f"[INFO] Model initialized successfully")
        print(f"  - Input shape: {self.input_shape}")
        print(f"  - Flattened size: {self.flattened_size}")
        print(f"  - Output classes: {self.num_classes}")
        print(f"  - Total parameters: ~{self._count_parameters():,}")

    def _count_parameters(self):
        """
        Đếm tổng số parameters trong model.
        
        Returns:
            int: Tổng số parameters
        """
        conv_params = self.conv_filters * self.conv_kernel_size * self.conv_kernel_size
        dense1_params = self.flattened_size * self.dense1_nodes + self.dense1_nodes
        dense2_params = self.dense1_nodes * self.num_classes + self.num_classes
        return conv_params + dense1_params + dense2_params

    def forward(self, image, training=True):
        """
        Forward pass qua toàn bộ network.
        
        Args:
            image (np.array): Ảnh đầu vào (height x width)
            training (bool): True nếu đang training (áp dụng dropout)
            
        Returns:
            tuple: (probabilities, logits)
                - probabilities: Xác suất cho mỗi class (tổng = 1)
                - logits: Raw scores trước softmax
        """
        # Normalize ảnh về khoảng [-0.5, 0.5]
        # Giúp training ổn định hơn
        img_processed = image - 0.5

        # Pass qua từng layer
        out = self.conv1.forward(img_processed)      # Conv
        out = self.relu1.forward(out)                # ReLU
        out = self.pool1.forward(out)                # MaxPool
        out = self.flatten_layer.forward(out)        # Flatten 3D -> 1D
        out = self.dense1.forward(out)               # Dense layer 1
        out = self.relu2.forward(out)                # ReLU
        out = self.dropout1.forward(out, training=training)  # Dropout (chỉ khi training)
        logits = self.dense2.forward(out)            # Output layer
        probs = self.softmax_activation.forward(logits)  # Softmax
        
        return probs, logits

    def predict(self, image):
        """
        Dự đoán class cho 1 ảnh (inference mode).
        
        Args:
            image (np.array): Ảnh đầu vào
            
        Returns:
            tuple: (predicted_class, probabilities)
        """
        probs, _ = self.forward(image, training=False)
        predicted_class = np.argmax(probs)
        return predicted_class, probs

    def save_model(self, path, idx_to_class_map):
        """
        Lưu model và các parameters vào file.
        
        Args:
            path (str): Đường dẫn file .npz để lưu
            idx_to_class_map (dict): Mapping từ index -> tên class
            
        Returns:
            bool: True nếu lưu thành công, False nếu thất bại
        """
        print(f"[INFO] Saving model to {path}...")
        
        try:
            # Tạo thư mục nếu chưa tồn tại
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

            # Tổ chức tất cả parameters cần lưu
            params_to_save = {
                # ===== Architecture hyperparameters =====
                'img_h': self.input_shape[0],
                'img_w': self.input_shape[1],
                'num_model_classes': self.num_classes,
                'conv_filters_count': self.conv_filters,
                'conv_kernel_size': self.conv_kernel_size,
                'pool_size': self.pool_size,
                'pool_stride': self.pool_stride,
                'dense1_nodes': self.dense1_nodes,
                'dropout_rate': self.dropout_rate,

                # ===== Trained weights and biases =====
                'conv1_filters': self.conv1.filters,
                'dense1_weights': self.dense1.weights,
                'dense1_biases': self.dense1.biases,
                'dense2_weights': self.dense2.weights,
                'dense2_biases': self.dense2.biases,
                
                # ===== Class mapping =====
                'idx_to_class_map': np.array(list(idx_to_class_map.items()), dtype=object)
            }

            # Lưu vào file .npz
            np.savez(path, **params_to_save)
            print(f"[SUCCESS] Model đã được lưu thành công: {path}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Lỗi khi lưu model: {e}")
            import traceback
            traceback.print_exc()
            return False

    @staticmethod
    def load_model(path):
        """
        Load model từ file đã lưu.
        
        Args:
            path (str): Đường dẫn đến file .npz
            
        Returns:
            tuple: (model, idx_to_class_map)
                - model: CNNModel instance với weights đã load
                - idx_to_class_map: Dictionary mapping index -> class name
                Trả về (None, None) nếu load thất bại
        """
        print(f"[INFO] Loading model from {path}...")
        
        # Kiểm tra file tồn tại
        if not os.path.exists(path):
            print(f"[ERROR] Model file không tồn tại: {path}")
            return None, None

        try:
            # Load file .npz
            data = np.load(path, allow_pickle=True)
            
            # Kiểm tra các keys bắt buộc
            required_keys = ['img_h', 'img_w', 'num_model_classes', 'conv_filters_count',
                           'conv_kernel_size', 'pool_size', 'pool_stride', 'dense1_nodes',
                           'dropout_rate', 'conv1_filters', 'dense1_weights', 'dense1_biases',
                           'dense2_weights', 'dense2_biases', 'idx_to_class_map']
            
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                print(f"[ERROR] File thiếu các keys: {missing_keys}")
                return None, None

            # ===== Khởi tạo model với architecture từ file =====
            model = CNNModel(
                input_shape=(int(data['img_h']), int(data['img_w'])),
                num_classes=int(data['num_model_classes']),
                conv_filters=int(data['conv_filters_count']),
                conv_kernel_size=int(data['conv_kernel_size']),
                pool_size=int(data['pool_size']),
                pool_stride=int(data['pool_stride']),
                dense1_nodes=int(data['dense1_nodes']),
                dropout_rate=float(data['dropout_rate'])
            )

            # ===== Load trained weights =====
            model.conv1.filters = data['conv1_filters']
            model.dense1.weights = data['dense1_weights']
            model.dense1.biases = data['dense1_biases']
            model.dense2.weights = data['dense2_weights']
            model.dense2.biases = data['dense2_biases']

            # ===== Load class mapping =====
            idx_to_class_map = {int(item[0]): item[1] for item in data['idx_to_class_map']}
            
            print("[SUCCESS] Model loaded successfully")
            return model, idx_to_class_map
            
        except KeyError as e:
            print(f"[ERROR] Missing key {e} trong model file")
            print("[INFO] Đảm bảo file được lưu với đầy đủ architecture params và weights")
            import traceback
            traceback.print_exc()
            return None, None
            
        except Exception as e:
            print(f"[ERROR] Lỗi khi load model: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def summary(self):
        """
        In ra summary của model architecture.
        """
        print("\n" + "="*60)
        print("MODEL SUMMARY".center(60))
        print("="*60)
        print(f"Input Shape: {self.input_shape}")
        print