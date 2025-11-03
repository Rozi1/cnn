import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
# ==========================================
# 6. Load model (optional reuse)
# ==========================================
model = load_model("mnist_cnn.h5")

# ==========================================
# 7. Function to test on a real image
# ==========================================
def predict_digit(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠️ Could not read {image_path}")
        return
    
    # Resize to 28x28
    img_resized = cv2.resize(img, (28, 28))
    
    # Invert if background is dark
    if np.mean(img_resized) < 127:
        img_resized = 255 - img_resized
    
    # Normalize
    img_resized = img_resized / 255.0
    
    # Reshape to match model input
    img_input = img_resized.reshape(1, 28, 28, 1)
    
    # Predict
    prediction = model.predict(img_input)
    print(f"Prediction probabilities: {prediction}")
    digit = np.argmax(prediction)
    
    # Display result
    plt.imshow(img_resized, cmap='gray')
    plt.title(f"Predicted Digit: {digit}")
    plt.axis('off')
    plt.show()
    print(f"🧩 Predicted Digit for {os.path.basename(image_path)}: {digit}")

# ==========================================
# 8. Test on your own image(s)
# ==========================================
# Place your image(s) in the same folder (e.g., 'digit.png')
# Make sure they are 28x28 or will be resized automatically.

predict_digit("digit.png")  # change filename if needed
# predict_digit("digit2.png")  # you can test multiple images
