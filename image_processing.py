import cv2
import numpy as np

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=-1)  # Add the channel dimension
    return img
