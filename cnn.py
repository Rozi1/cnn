# ==========================================
# 1. Import required libraries
# ==========================================
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# ==========================================
# 2. Load and preprocess MNIST data
# ==========================================
(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

# Normalize (0–255 → 0–1)
X_train, X_test = X_train / 255.0, X_test / 255.0

# Reshape for CNN (28x28x1)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# ==========================================
# 3. Build a simple CNN model
# ==========================================
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ==========================================
# 4. Train the model
# ==========================================
print("\nTraining model...")
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# Evaluate accuracy
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test accuracy: {test_acc:.4f}")

# ==========================================
# 5. Save the trained model
# ==========================================
model.save("mnist_cnn.h5")
print("\n💾 Model saved as 'mnist_cnn.h5'")

