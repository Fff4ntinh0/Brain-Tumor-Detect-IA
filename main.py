import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import os
import cv2
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model

dataset_location = '/home/fffantinho/Programing/Brain_Tumor/brain_tumor_dataset'
image_size = 128
modelo_path = "modelo_tumor_cnn.h5"

def load_dataset(path):
    images=[]
    results=[]
    for classe in ['no', 'yes']:
        way_class = os.path.join(path, classe)
        for file_name in os.listdir(way_class):
            way_image = os.path.join(way_class, file_name)
            image = cv2.imread(way_image, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                image = cv2.resize(image,(image_size, image_size))
                image = image/255.0
                images.append(image)
                results.append(classe)
    return np.array(images), np.array(results)

def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 1)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

def evaluate_model(model, X_test, y_test, encoder):
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

def save_model(model, path=modelo_path):
    model.save(path)

def load_trained_model(path=modelo_path):
    try:
        model = load_model(path)
        print("Model sucess load!")
        return model
    except Exception as e:
        print("Model not found:", e)
        return None

def predict_image(model, image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: image not found.")
        return None
    image = cv2.resize(image, (image_size, image_size))
    image = image / 255.0
    image = image.reshape(1, image_size, image_size, 1)
    pred = model.predict(image)[0][0]
    return pred

def painel():
    print("\n=== Painel de Controle - Detecção de Tumor Cerebral ===")
    print("1 - Carregar dataset")
    print("2 - Criar modelo")
    print("3 - Treinar modelo")
    print("4 - Avaliar modelo")
    print("5 - Salvar modelo")
    print("6 - Carregar modelo salvo")
    print("7 - Prever imagem nova")
    print("8 - Rodar tudo (carregar, criar, treinar, avaliar, salvar)")
    print("0 - Sair")

def show_menu():
    print("\n=== Brain Tumor Detection Control Panel ===")
    print("1 - Load dataset")
    print("2 - Create model")
    print("3 - Train model")
    print("4 - Evaluate model")
    print("5 - Save model")
    print("6 - Load saved model")
    print("7 - Predict on new image")
    print("8 - Run full pipeline (load, create, train, evaluate, save)")
    print("0 - Exit")

if __name__ == "__main__":
    model = None
    X, y = None, None
    X_train, X_test, y_train, y_test = None, None, None, None
    encoder = None

    while True:
        show_menu()
        choice = input("Choose an option: ")

        if choice == "1":
            X, y = load_dataset(dataset_location)
            X = X.reshape(-1, image_size, image_size, 1)
            encoder = LabelEncoder()
            y_encoded = encoder.fit_transform(y)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42
            )
            print("Dataset loaded and split.")

        elif choice == "2":
            model = create_model()
            print("Model created.")

        elif choice == "3":
            if model is None or X_train is None or y_train is None:
                print("Please load the dataset and create the model first.")
            else:
                train_model(model, X_train, y_train)
                print("Training completed.")

        elif choice == "4":
            if (
                model is None
                or X_test is None
                or y_test is None
                or encoder is None
            ):
                print("Please load data, create and train the model first.")
            else:
                evaluate_model(model, X_test, y_test, encoder)

        elif choice == "5":
            if model is None:
                print("Please create or load a model first.")
            else:
                save_model(model)
                print(f"Model saved to {modelo_path}.")

        elif choice == "6":
            model = load_trained_model()

        elif choice == "7":
            if model is None:
                print("Please load or create and train a model first.")
            else:
                image_path = input("Enter the path of the image to predict: ")
                probability = predict_image(model, image_path)
                if probability is not None:
                    print(f"Tumor probability for the image: {probability:.4f}")

        elif choice == "8":
            X, y = load_dataset(dataset_location)
            X = X.reshape(-1, image_size, image_size, 1)
            encoder = LabelEncoder()
            y_encoded = encoder.fit_transform(y)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42
            )
            model = create_model()
            train_model(model, X_train, y_train)
            evaluate_model(model, X_test, y_test, encoder)
            save_model(model)
            print("Full pipeline executed successfully.")

        elif choice == "0":
            print("Exiting...")
            break

        else:
            print("Invalid option. Please try again.")
