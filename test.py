import numpy as np
import cv2
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("model/malaria_model.h5")

# Read test image
img = cv2.imread("test.jpg")

# Resize image
img = cv2.resize(img, (128,128))

# Normalize
img = img / 255.0

# Reshape for model
img = img.reshape(1,128,128,3)

# Prediction
prediction = model.predict(img)

if prediction > 0.5:
    print("Parasitized (Malaria Detected)")
else:
    print("Uninfected")