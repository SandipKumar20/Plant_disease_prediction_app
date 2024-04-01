import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

# Loading the pretrained neural net

model = load_model("plant_disease.h5")

# Name of Classes
CLASS_NAMES = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

# Setting the title of the app
st.title("Plant Disease Prediction")
st.markdown("Upload an image of a plant leaf")

# Uploading an image of a plant leaf
plant_image = st.file_uploader("Choose an image...")
submit = st.button("Predict")

if submit:
    if plant_image is not None:

        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Displaying the image
        st.image(opencv_image, channels="BGR")
        st.write(opencv_image.shape)

        # Resizing the image
        opencv_image = cv2.resize(opencv_image, (256, 256))

        # Convert the image to 4 dimensions
        opencv_image.shape = (1, 256, 256, 3)

        # Make prediction
        y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(y_pred)]
        st.header("This is a " + result.split("-")[0] + " leaf with " + result.split("-")[1] + " disease.")


