import os
import cv2
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

def load_and_preprocess_images(images):
    images, labels = [], []
    for pothole_1.jpg in os.listdir(images):
        img = cv2.imread(os.path.join(images, pothole_1.jpg))
        if img is not None:
            img = cv2.resize(img, (224, 224))
            images.append(img)
            labels.append(1 if 'pothole' in pothole_3.jpg else 0)
    return np.array(images), np.array(labels)

def create_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    datagen = ImageDataGenerator()
    train_gen = datagen.flow(X_train, y_train, batch_size=32)
    val_gen = datagen.flow(X_val, y_val, batch_size=32)
    history = model.fit(train_gen, validation_data=val_gen, epochs=10)
    return history.history['accuracy'][-1]

def detect_and_label_images(model, images, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    results = []
    for i, img in enumerate(images):
        pred = model.predict(img.reshape(1, 224, 224, 3))[0][0]
        label = "pothole" if pred > 0.5 else "no_pothole"
        cv2.rectangle(img, (10, 10), (214, 214), (255, 0, 0), 2)
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        output_path = os.path.join(output_folder, f"image_{i}.jpg")
        cv2.imwrite(output_path, img)
        results.append([output_path, int(pred > 0.5)])
    return results

def pothole_detection(images):
    images, labels = load_and_preprocess_images(images)
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2)
    model = create_model()
    accuracy = train_model(model, X_train, y_train, X_val, y_val)
    print(f"Model trained with accuracy: {accuracy*100:.2f}%")
    results = detect_and_label_images(model, images, 'Label_image')
    df = pd.DataFrame(results, columns=['file_name', 'pothole_count'])
    df.to_csv('pothole_detection_results.csv', index=False)

# Example call to the function (uncomment to use in real scenarios):
# pothole_detection('path_to_image_folder')
