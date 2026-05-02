# =========================
# LOAD MODEL & PREDICT
# =========================

import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# LOAD SAVED MODEL
model = load_model("mnist_cnn_model.h5")

print("✅ Model loaded successfully")

# LOAD IMAGE (CHANGE PATH)
img_path = r"C:\Users\antar\OneDrive\Desktop\WhatsApp Image 2026-05-02 at 4.01.19 PM.jpeg"

# READ IMAGE
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# RESIZE TO 28x28
img = cv2.resize(img, (28, 28))

# INVERT IMAGE (if background is white)
img = 255 - img

# NORMALIZE
img = img / 255.0

# RESHAPE
img = img.reshape(1, 28, 28, 1)

# PREDICT
prediction = model.predict(img)
digit = np.argmax(prediction)

print(f"🧠 Predicted Digit: {digit}")

# SHOW IMAGE
plt.imshow(img.reshape(28, 28), cmap='gray')
plt.title(f"Prediction: {digit}")
plt.axis('off')
plt.show()