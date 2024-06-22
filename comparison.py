import tensorflow as tf
import numpy as np
from image_processing import preprocess_image  # Ensure we use the same preprocessing
import cv2

@tf.keras.utils.register_keras_serializable()
def euclidean_distance(vects):
    x, y = vects
    return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True))

@tf.keras.utils.register_keras_serializable()
def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

@tf.keras.utils.register_keras_serializable()
def contrastive_loss(y_true, y_pred):
    margin = 1
    return tf.keras.backend.mean(y_true * tf.keras.backend.square(y_pred) + (1 - y_true) * tf.keras.backend.square(tf.keras.backend.maximum(margin - y_pred, 0)))

custom_objects = {
    'contrastive_loss': contrastive_loss,
    'euclidean_distance': euclidean_distance,
    'eucl_dist_output_shape': eucl_dist_output_shape
}
model = tf.keras.models.load_model('final_model.keras', custom_objects=custom_objects)

def preprocess_image_for_comparison(image):
    if isinstance(image, str):
        return preprocess_image(image)  # Path-based preprocessing
    else:
        # Handle image arrays
        image = np.array(image)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (128, 128))
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=-1)  # Add the channel dimension
        return image

def compare_features_cnn(genuine_features, suspicious_features):
    genuine_features = np.array(genuine_features).reshape(-1, 128, 128, 1)
    suspicious_features = np.expand_dims(np.array(suspicious_features).reshape(128, 128, 1), axis=0)
    
    # Debug: Check shapes and values of features
    print("Genuine Features Shape:", genuine_features.shape)
    print("Suspicious Features Shape:", suspicious_features.shape)
    print("Genuine Features (first):", genuine_features[0])
    print("Suspicious Features:", suspicious_features[0])
    
    distances = model.predict([genuine_features, suspicious_features])
    
    # Debug: Check the distances
    print("Distances:", distances)
    
    similarity_scores = 100 - (distances * 100)
    
    average_similarity_score = np.mean(similarity_scores)
    
    return average_similarity_score

def validate_signature_cnn(genuine, suspicious):
    genuine_features = [preprocess_image_for_comparison(img) for img in genuine]
    suspicious_features = preprocess_image_for_comparison(suspicious)
    return compare_features_cnn(genuine_features, suspicious_features)
