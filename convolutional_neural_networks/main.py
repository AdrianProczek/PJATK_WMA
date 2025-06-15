import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# === Parametry ===
IMG_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 3

# === Ścieżki do danych ===
TRAIN_DIR = 'fruits/train'
TEST_DIR = 'fruits/test'

# === Augmentacja i ładowanie danych ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# === Model CNN ===
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# === Trenowanie modelu ===
model.fit(train_data, epochs=EPOCHS, validation_data=test_data)

# === Ewaluacja ===
loss, acc = model.evaluate(test_data)
print(f"Test accuracy: {acc:.2f}")

# === Zapis modelu ===
model.save("fruit_classifier_model.h5")
print("Model saved as fruit_classifier_model.h5")

# === Klasy dla przewidywania ===
class_labels = list(train_data.class_indices.keys())

# === Przykład testowania pojedynczego obrazu ===
def predict_single_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0]
    class_idx = np.argmax(pred)
    print(f"Prediction: {class_labels[class_idx]} ({pred[class_idx]*100:.2f}%)")


# predict_single_image("sciezka do przykaldowego obrazka")
