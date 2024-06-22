import cv2

def extract_features(image_path):
    if isinstance(image_path, str):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")
    else:
        image = image_path  
    
    image = cv2.resize(image, (100, 100))
    image = image.astype('float32') / 255.0
    return image
