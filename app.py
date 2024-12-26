import streamlit as st
import os
import PIL
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.layers import MaxPool2D, BatchNormalization, Dropout

# Constants for file paths and image processing
UPLOAD_PATH = "./website/static/uploads/"
WEIGHTS_PATH = "./website/static/model_weights/model-050.hdf5"

IMG_RES = {
    "resize": (28, 28),
    "input_shape": (28, 28, 3),
    "reshape": (-1, 28, 28, 3)
}

CLASSES = {
    0: "actinic keratoses and intraepithelial carcinomae (Cancer)",
    1: "basal cell carcinoma (Cancer)",
    2: "benign keratosis-like lesions (Non-Cancerous)",
    3: "dermatofibroma (Non-Cancerous)",
    4: "melanocytic nevi (Non-Cancerous)",
    5: "pyogenic granulomas and hemorrhage (Can lead to cancer)",
    6: "melanoma (Cancer)"
}

# Function to create the CNN model
def create_model():
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3,3), input_shape=(28, 28, 3), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
    model.add(Flatten())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(Dense(256,activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(Dense(128,activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(Dense(64,activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(Dense(32,activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(Dense(7,activation='softmax'))


    Optimizer = Adam(learning_rate=0.001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Optimizer, metrics=['accuracy'])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Define the callback with the updated file extension
    model.load_weights(WEIGHTS_PATH)
    return model

# Load the model
MODEL = create_model()

# Function to predict the class of the uploaded image
def predict(filename):
    image = PIL.Image.open(os.path.join(UPLOAD_PATH, filename))
    image = image.resize(IMG_RES["resize"])
    image = np.array(image).reshape(IMG_RES["reshape"])
    
    prediction = MODEL.predict(image)[0]
    prediction = sorted(
        [(CLASSES[i], round(j * 100, 2)) for i, j in enumerate(prediction)],
        reverse=True,
        key=lambda x: x[1]
    )
    return prediction

# Streamlit UI components
st.title("Skin Disease Detector")
st.write("Welcome to the Skin Disease Detector!")
st.write("Please upload an image for analysis.")

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # To read the file as bytes:
    file_bytes = uploaded_file.read()
    
    # Save the uploaded file to a specified path
    upload_path = os.path.join(UPLOAD_PATH, uploaded_file.name)
    with open(upload_path, "wb") as f:
        f.write(file_bytes)
    
    # Display the uploaded image
    st.image(file_bytes, caption='Uploaded Image', use_column_width=True)
    
    # Classifying the image
    st.write("Classifying the image...")
    prediction = predict(uploaded_file.name)  # Use the predict function

    # Display prediction results
    if prediction:
        st.write("Prediction Results:")
        st.write("Here are the top predictions:")
        for class_name, confidence in prediction:
            st.write(f"{class_name}: {confidence}%")
