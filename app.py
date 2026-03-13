import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("model/malaria_model.h5")

st.title("Malaria Detection using Deep Learning")

st.write("Upload a blood cell image to detect malaria.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = cv2.resize(img, (128,128))
    img = img / 255.0
    img = img.reshape(1,128,128,3)

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        st.success("Uninfected Cell")
    else:
        st.error("Parasitized Cell (Malaria Detected)")