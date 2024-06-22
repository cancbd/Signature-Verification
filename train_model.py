import tensorflow as tf
import numpy as np
import pandas as pd
import os
import datetime
from PIL import Image
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Lambda, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from sklearn.utils import shuffle
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda, Dropout, BatchNormalization
import cv2
from tqdm import tqdm
from tensorflow.keras import regularizers

df = pd.read_csv('data/train_data.csv')
path = 'data/train/'

x = []
y = []

for ind in tqdm(df.index):
    name1 = df['068/09_068.png'][ind]
    name2 = df['068_forg/03_0113068.PNG'][ind]
    
    img1_path = os.path.join(path, str(name1))
    img2_path = os.path.join(path, str(name2))
    
    if os.path.exists(img1_path) and os.path.exists(img2_path):
        img1 = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2GRAY)
        
        img1 = np.array(img1).astype('float32')/255
        img2 = np.array(img2).astype('float32')/255
        
        img1 = cv2.resize(img1, (128, 128), cv2.INTER_CUBIC)
        img2 = cv2.resize(img2, (128, 128), cv2.INTER_CUBIC)
        
        x += [[img1, img2]]
        y += [df['1'][ind]]
    else:
        print(f"Image path does not exist: {img1_path} or {img2_path}")

x = np.array(x)
y = np.array(y)
x_train1 = x[:17000,0]
x_train2 = x[:17000,1]

def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

input = layers.Input((128, 128, 1))
x = tf.keras.layers.BatchNormalization()(input)
x = layers.Conv2D(32, (5, 5), activation="relu", kernel_regularizer=regularizers.L2(l2=2e-4),
    bias_regularizer=regularizers.L2(2e-4))(x)
x = layers.AveragePooling2D(pool_size=(2, 2))(x)
x = layers.Dropout(0.3)(x)
x = layers.Conv2D(32, (5, 5), kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.L2(2e-4), activation="relu")(x)
x = layers.AveragePooling2D(pool_size=(2, 2))(x)
x = layers.Dropout(0.3)(x)
x = layers.Flatten()(x)

x = tf.keras.layers.BatchNormalization()(x)
x = layers.Dense(128, activation="relu")(x)
embedding_network = tf.keras.Model(input, x)

input_1 = layers.Input((128, 128, 1))
input_2 = layers.Input((128, 128, 1))

tower_1 = embedding_network(input_1)
tower_2 = embedding_network(input_2)

merge_layer = layers.Lambda(euclidean_distance)([tower_1, tower_2])
normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
output_layer = layers.Dense(1, activation="sigmoid")(normal_layer)
siamese = tf.keras.Model(inputs=[input_1, input_2], outputs=output_layer)

# Compile the model
siamese.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

model_checkpoint_callback = ModelCheckpoint('final_model.keras', save_best_only=True, verbose=1)

# Fit the model
siamese.fit(
    [x_train1, x_train2],
    y,
    validation_split=0.1,
    batch_size=32,
    epochs=10,
    verbose=1,
    callbacks=[model_checkpoint_callback]
)
