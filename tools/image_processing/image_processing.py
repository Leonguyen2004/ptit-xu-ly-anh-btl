import cv2
import numpy as np

def to_grayscale(image):
    """Convert image to grayscale."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    """
    Adjust brightness and contrast.
    brightness: -127 to 127
    contrast: -127 to 127
    """
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        
        buf = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)
    else:
        buf = image.copy()
    
    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    
    return buf

def gaussian_blur(image, kernel_size=5):
    """Apply Gaussian Blur."""
    # Kernel size must be odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def median_blur(image, kernel_size=5):
    """Apply Median Blur."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(image, kernel_size)

def canny_edge_detection(image, threshold1=100, threshold2=200):
    """Apply Canny Edge Detection."""
    return cv2.Canny(image, threshold1, threshold2)

def threshold_otsu(image):
    """Apply Otsu's Binarization."""
    gray = to_grayscale(image)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def threshold_adaptive(image, block_size=11, C=2):
    """Apply Adaptive Thresholding."""
    gray = to_grayscale(image)
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, block_size, C)
