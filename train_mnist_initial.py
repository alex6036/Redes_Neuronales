import tensorflow as tf
import numpy as np
import os

MODEL_PATH = "models/mnist_model.h5"
DATA_PATH = "data"
os.makedirs("models", exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)

# -----------------------
# Cargar MNIST
# -----------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizar y aplanar imágenes
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0

# One-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# -----------------------
# Guardar dataset base para reentrenamiento futuro
# -----------------------
np.savez(os.path.join(DATA_PATH, "mnist_full.npz"), X=x_train, y=y_train)

# -----------------------
# Crear y entrenar modelo
# -----------------------
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation="relu", input_shape=(28*28,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Guardar modelo
model.save(MODEL_PATH)
