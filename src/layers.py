import numpy as np

class Conv:
    """
    Convolutional Layer - Áp dụng các bộ lọc (filters) lên ảnh đầu vào.
    
    Args:
        num_filters (int): Số lượng bộ lọc (filters)
        filter_size (int): Kích thước bộ lọc (mặc định 3x3)
    """
    def __init__(self, num_filters, filter_size=3):
        self.num_filters = num_filters
        self.filter_size = filter_size
        # Khởi tạo filters với phân phối chuẩn và scale theo kích thước
        # Giúp tránh vanishing/exploding gradients
        self.filters = np.random.randn(num_filters, filter_size, filter_size) / (filter_size * filter_size)
        self.padding = 0

    def iterate_regions(self, image):
        """
        Generator để duyệt qua các vùng của ảnh với kích thước filter.
        
        Args:
            image (np.array): Ảnh đầu vào 2D
            
        Yields:
            tuple: (vùng ảnh, chỉ số hàng i, chỉ số cột j)
        """
        h, w = image.shape
        # Duyệt qua từng vị trí có thể đặt filter (valid convolution)
        for i in range(h - self.filter_size + 1):
            for j in range(w - self.filter_size + 1):
                # Trích xuất vùng ảnh có kích thước bằng filter
                im_region = image[i:(i + self.filter_size), j:(j + self.filter_size)]
                yield im_region, i, j

    def forward(self, input_data):
        """
        Forward pass: Áp dụng convolution lên ảnh đầu vào.
        
        Args:
            input_data (np.array): Ảnh đầu vào 2D (h x w)
            
        Returns:
            np.array: Feature maps 3D (h' x w' x num_filters)
        """
        # Lưu input để dùng cho backpropagation
        self.last_input = input_data
        h, w = input_data.shape
        
        # Tính kích thước output (valid convolution, không padding)
        output_h = h - self.filter_size + 1
        output_w = w - self.filter_size + 1
        output = np.zeros((output_h, output_w, self.num_filters))

        # Áp dụng mỗi filter lên từng vùng ảnh
        for im_region, i, j in self.iterate_regions(input_data):
            # Tính tích element-wise rồi sum theo 2 chiều không gian
            # Kết quả: vector có num_filters phần tử
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
        
        return output

    def backprop(self, d_l_d_out, learn_rate):
        """
        Backward pass: Tính gradient và cập nhật filters.
        
        Args:
            d_l_d_out (np.array): Gradient từ layer sau (h' x w' x num_filters)
            learn_rate (float): Learning rate
            
        Returns:
            None: Layer này không truyền gradient về trước (vì là layer đầu tiên)
        """
        # Khởi tạo gradient cho filters
        d_l_d_filters = np.zeros(self.filters.shape)
        
        # Duyệt qua từng vùng ảnh đã xử lý trong forward pass
        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                # Gradient của filter = gradient output * input region
                # (Chain rule: dL/df = dL/dout * dout/df)
                d_l_d_filters[f] += d_l_d_out[i, j, f] * im_region
        
        # Cập nhật filters bằng gradient descent
        self.filters -= learn_rate * d_l_d_filters
        return None


class ReLU:
    """
    ReLU Activation Layer - Áp dụng hàm kích hoạt ReLU(x) = max(0, x).
    """
    def forward(self, input_data):
        """
        Forward pass: Áp dụng ReLU activation.
        
        Args:
            input_data (np.array): Input bất kỳ shape
            
        Returns:
            np.array: Output cùng shape, với giá trị âm thành 0
        """
        # Lưu input để tính gradient
        self.last_input = input_data
        # ReLU: giữ giá trị dương, chuyển âm thành 0
        return np.maximum(0, input_data)

    def backprop(self, d_l_d_out):
        """
        Backward pass: Tính gradient qua ReLU.
        
        Args:
            d_l_d_out (np.array): Gradient từ layer sau
            
        Returns:
            np.array: Gradient truyền về trước (gradient = 0 nếu input <= 0)
        """
        # Gradient của ReLU: 1 nếu input > 0, 0 nếu input <= 0
        return d_l_d_out * (self.last_input > 0)


class MaxPool:
    """
    Max Pooling Layer - Giảm kích thước không gian bằng cách lấy giá trị max.
    
    Args:
        pool_size (int): Kích thước pooling window (mặc định 2x2)
        stride (int): Bước nhảy khi di chuyển window (mặc định 2)
    """
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def iterate_regions(self, image):
        """
        Generator để duyệt qua các vùng pooling.
        
        Args:
            image (np.array): Input 3D (h x w x num_filters)
            
        Yields:
            tuple: (vùng pooling, chỉ số output i, chỉ số output j)
        """
        h, w, num_filters = image.shape
        # Tính kích thước output
        new_h = (h - self.pool_size) // self.stride + 1
        new_w = (w - self.pool_size) // self.stride + 1
        
        for i in range(new_h):
            for j in range(new_w):
                # Trích xuất vùng pooling theo stride
                im_region = image[(i * self.stride):(i * self.stride + self.pool_size),
                                  (j * self.stride):(j * self.stride + self.pool_size)]
                yield im_region, i, j

    def forward(self, input_data):
        """
        Forward pass: Áp dụng max pooling.
        
        Args:
            input_data (np.array): Input 3D (h x w x num_filters)
            
        Returns:
            np.array: Output 3D với kích thước giảm (h' x w' x num_filters)
        """
        # Lưu input cho backprop
        self.last_input = input_data
        h, w, num_filters = input_data.shape
        
        # Tính kích thước output
        output_h = (h - self.pool_size) // self.stride + 1
        output_w = (w - self.pool_size) // self.stride + 1
        output = np.zeros((output_h, output_w, num_filters))

        # Lấy giá trị max từ mỗi vùng pooling
        for im_region, i, j in self.iterate_regions(input_data):
            # Max theo 2 chiều không gian (height, width), giữ nguyên channels
            output[i, j] = np.amax(im_region, axis=(0, 1))
        
        return output

    def backprop(self, d_l_d_out):
        """
        Backward pass: Truyền gradient chỉ về vị trí có giá trị max.
        
        Args:
            d_l_d_out (np.array): Gradient từ layer sau (h' x w' x num_filters)
            
        Returns:
            np.array: Gradient cho input (h x w x num_filters)
        """
        # Khởi tạo gradient input với 0
        d_l_d_input = np.zeros(self.last_input.shape)
        
        for im_region, i, j in self.iterate_regions(self.last_input):
            h_r, w_r, num_filters_r = im_region.shape
            amax = np.amax(im_region, axis=(0, 1))
            
            # Tìm vị trí có giá trị max và gán gradient vào đó
            for r_i in range(h_r):
                for r_j in range(w_r):
                    for f_k in range(num_filters_r):
                        # Chỉ vị trí max mới nhận gradient
                        if im_region[r_i, r_j, f_k] == amax[f_k]:
                            # Chuyển đổi tọa độ local sang tọa độ global
                            input_i = i * self.stride + r_i
                            input_j = j * self.stride + r_j
                            d_l_d_input[input_i, input_j, f_k] += d_l_d_out[i, j, f_k]
        
        return d_l_d_input


class Flatten:
    """
    Flatten Layer - Chuyển đổi từ tensor nhiều chiều thành vector 1D.
    """
    def forward(self, input_data):
        """
        Forward pass: Làm phẳng input thành 1D.
        
        Args:
            input_data (np.array): Input nhiều chiều
            
        Returns:
            np.array: Vector 1D
        """
        # Lưu shape để reshape lại khi backprop
        self.last_input_shape = input_data.shape
        return input_data.flatten()

    def backprop(self, d_l_d_out):
        """
        Backward pass: Reshape gradient về shape ban đầu.
        
        Args:
            d_l_d_out (np.array): Gradient 1D
            
        Returns:
            np.array: Gradient với shape giống input
        """
        return d_l_d_out.reshape(self.last_input_shape)


class Dense:
    """
    Fully Connected (Dense) Layer - Linear transformation: y = Wx + b.
    
    Args:
        input_len (int): Số lượng neurons đầu vào
        output_len (int): Số lượng neurons đầu ra
    """
    def __init__(self, input_len, output_len):
        # Khởi tạo weights với Xavier initialization
        # Scale theo sqrt(input_len) giúp ổn định training
        self.weights = np.random.randn(input_len, output_len) / np.sqrt(input_len)
        # Khởi tạo biases = 0
        self.biases = np.zeros(output_len)

    def forward(self, input_data):
        """
        Forward pass: y = Wx + b.
        
        Args:
            input_data (np.array): Vector đầu vào (input_len,)
            
        Returns:
            np.array: Vector đầu ra (output_len,)
        """
        # Lưu input cho backprop
        self.last_input = input_data
        # Linear transformation
        return np.dot(input_data, self.weights) + self.biases

    def backprop(self, d_l_d_out, learn_rate):
        """
        Backward pass: Tính gradients và cập nhật weights, biases.
        
        Args:
            d_l_d_out (np.array): Gradient từ layer sau (output_len,)
            learn_rate (float): Learning rate
            
        Returns:
            np.array: Gradient cho input (input_len,)
        """
        # Gradient cho weights: dL/dW = input^T * dL/dout (outer product)
        d_l_d_weights = np.outer(self.last_input, d_l_d_out)
        # Gradient cho biases: dL/db = dL/dout
        d_l_d_biases = d_l_d_out
        # Gradient cho input: dL/dinput = dL/dout * W^T
        d_l_d_input = np.dot(d_l_d_out, self.weights.T)
        
        # Cập nhật parameters
        self.weights -= learn_rate * d_l_d_weights
        self.biases -= learn_rate * d_l_d_biases
        
        return d_l_d_input


class Dropout:
    """
    Dropout Layer - Regularization bằng cách randomly tắt neurons.
    
    Args:
        rate (float): Tỉ lệ neurons bị tắt (0.0 - 1.0)
    """
    def __init__(self, rate):
        if not 0.0 <= rate < 1.0:
            raise ValueError(f"Dropout rate phải trong khoảng [0, 1), nhận được: {rate}")
        self.rate = rate
        self.mask = None

    def forward(self, input_data, training=True):
        """
        Forward pass: Áp dụng dropout khi training.
        
        Args:
            input_data (np.array): Input bất kỳ shape
            training (bool): True nếu đang training, False nếu inference
            
        Returns:
            np.array: Output sau khi áp dụng dropout
        """
        if training:
            # Tạo mask: giữ lại (1 - rate) neurons
            # Scale lên để giữ expected value không đổi (inverted dropout)
            self.mask = (np.random.rand(*input_data.shape) > self.rate) / (1.0 - self.rate)
            return input_data * self.mask
        else:
            # Không dropout khi inference
            return input_data

    def backprop(self, d_l_d_out):
        """
        Backward pass: Áp dụng mask lên gradient.
        
        Args:
            d_l_d_out (np.array): Gradient từ layer sau
            
        Returns:
            np.array: Gradient đã được mask
        """
        # Gradient chỉ truyền qua neurons không bị dropout
        return d_l_d_out * self.mask


class Softmax:
    """
    Softmax Activation Layer - Chuyển logits thành xác suất.
    Output: probability distribution tổng = 1.
    """
    def forward(self, input_data):
        """
        Forward pass: Tính softmax với numerical stability.
        
        Args:
            input_data (np.array): Logits (raw scores)
            
        Returns:
            np.array: Probability distribution
        """
        self.last_input_logits = input_data
        
        # Numerical stability: trừ max để tránh overflow trong exp
        # exp(x - max) thay vì exp(x)
        exp_shifted = np.exp(input_data - np.max(input_data, axis=-1, keepdims=True))
        # Normalize để tổng = 1
        self.last_output_probs = exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)
        
        return self.last_output_probs

    def backprop(self, d_l_d_out_probs):
        """
        Backward pass: Tính gradient qua softmax.
        
        Công thức: dL/dz_k = p_k * (dL/dp_k - sum_i(dL/dp_i * p_i))
        
        Args:
            d_l_d_out_probs (np.array): Gradient theo probabilities
            
        Returns:
            np.array: Gradient theo logits
        """
        p = self.last_output_probs
        # Jacobian của softmax là phức tạp, công thức này tính hiệu quả
        dL_dlogits = p * (d_l_d_out_probs - np.sum(d_l_d_out_probs * p, axis=-1, keepdims=True))
        return dL_dlogits