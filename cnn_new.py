# ==========================================
# 1. Import required libraries
# ==========================================
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import matplotlib.pyplot as plt

# ==========================================
# 2. Set paths and parameters
# ==========================================
data_dir = "data"   # folder with class subfolders
img_size = (128, 128)
batch_size = 32
epochs = 20

# ==========================================
# 3. Data loading & augmentation
# ==========================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,         # 80% train, 20% validation
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

train_gen = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    subset='training',
    class_mode='sparse'
)

val_gen = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    subset='validation',
    class_mode='sparse'
)

# ==========================================
# 4. Build a CNN model
# ==========================================
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(*img_size, 3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(train_gen.class_indices), activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Show model summary
model.summary()

# ==========================================
# 5. Train the model
# ==========================================
print("\n🚀 Training model...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs
)

# ==========================================
# 6. Evaluate the model
# ==========================================
val_loss, val_acc = model.evaluate(val_gen, verbose=0)
print(f"\n✅ Validation Accuracy: {val_acc:.4f}")

# ==========================================
# 7. Save the trained model
# ==========================================
model.save("image_classifier.h5")
print("\n💾 Model saved as 'image_classifier.h5'")

# ==========================================
# 8. Plot accuracy curve
# ==========================================
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()
