import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import io
from comparison import validate_signature_cnn

# Load the trained model
@tf.keras.utils.register_keras_serializable()
def euclidean_distance(vects):
    x, y = vects
    return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True))

@tf.keras.utils.register_keras_serializable()
def contrastive_loss(y_true, y_pred):
    margin = 1
    return tf.keras.backend.mean(y_true * tf.keras.backend.square(y_pred) + (1 - y_true) * tf.keras.backend.square(tf.keras.backend.maximum(margin - y_pred, 0)))

model = tf.keras.models.load_model('final_model.keras', custom_objects={
    'contrastive_loss': contrastive_loss,
    'euclidean_distance': euclidean_distance
})

def preprocess_image(image):
    image = np.array(image)
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (128, 128))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=-1)  # Add the channel dimension
    return image

st.title("Signature Validator")

st.sidebar.header("Upload Signatures")
genuine_signature = st.sidebar.file_uploader("Upload Genuine Signature", type=["png", "jpg", "jpeg"])
suspicious_signature = st.sidebar.file_uploader("Upload Suspicious Signature", type=["png", "jpg", "jpeg"])

# Variables to store images
genuine_img = None
suspicious_img = None

# Reset button
if st.sidebar.button("Reset"):
    genuine_signature = None
    suspicious_signature = None
    st.experimental_rerun()

# Display uploaded images
if genuine_signature is not None:
    genuine_img = Image.open(io.BytesIO(genuine_signature.read()))
    st.image(genuine_img, caption="Genuine Signature", use_column_width=True)

if suspicious_signature is not None:
    suspicious_img = Image.open(io.BytesIO(suspicious_signature.read()))
    st.image(suspicious_img, caption="Suspicious Signature", use_column_width=True)

# Perform validation
if st.sidebar.button("Compare"):
    if genuine_signature is not None and suspicious_signature is not None:
        genuine_img = Image.open(io.BytesIO(genuine_signature.getvalue()))
        suspicious_img = Image.open(io.BytesIO(suspicious_signature.getvalue()))

        # Check if the file paths are identical
        if genuine_signature.name == suspicious_signature.name:
            st.write("The images are identical. Similarity percentage with genuine signature: 100%.")
        else:
            genuine_features = preprocess_image(genuine_img)
            suspicious_features = preprocess_image(suspicious_img)

            # Debug: Print preprocessed features
            print("Preprocessed Genuine Features:", genuine_features)
            print("Preprocessed Suspicious Features:", suspicious_features)

            similarity_percentage = validate_signature_cnn([genuine_features], suspicious_features)

            st.write(f"Similarity percentage with genuine signature: {similarity_percentage:.2f}%.")
    else:
        st.write("Please upload both signatures to compare.")

# Information about similarity percentages
st.sidebar.header("Similarity Information")
st.sidebar.write("""
- **0-20%**: Forged
- **20-40%**: Slightly Forged
- **40-60%**: Moderate Similarity
- **60-80%**: High Similarity
- **80-100%**: Genuine
""")
