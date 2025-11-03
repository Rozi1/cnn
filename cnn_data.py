import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Load model and define class labels
# ==========================================
model = tf.keras.models.load_model("image_classifier.h5")
class_names = ['bike', 'cars', 'cats', 'dogs', 'flowers', 'horses', 'human']

# ==========================================
# 2. Function to predict image class
# ==========================================
def predict_image(img_path, threshold=0.6):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    confidence = np.max(predictions[0])
    predicted_index = np.argmax(predictions[0])
    predicted_label = class_names[predicted_index]

    # Apply threshold for unknown detection
    if confidence < threshold:
        predicted_label = "unknown"

    # Print results
    print(f"\n🖼️ Image: {img_path}")
    print(f"🔮 Predicted class: {predicted_label}")
    print(f"🔢 Confidence: {confidence*100:.2f}%")

    # Show image
    plt.imshow(image.load_img(img_path))
    plt.title(f"Predicted: {predicted_label} ({confidence*100:.1f}%)")
    plt.axis("off")
    plt.show()

# ==========================================
# 3. Example usage
# ==========================================
# Change path to your test image
predict_image("test.jpg")
