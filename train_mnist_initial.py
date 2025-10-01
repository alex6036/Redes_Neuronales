import numpy as np
import tensorflow as tf
import os

DATA_PATH = "data"
MODEL_PATH = "models/mnist_model.h5"
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs("models", exist_ok=True)

# Cargar MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)

# Guardar dataset completo
np.savez(os.path.join(DATA_PATH, "mnist_full.npz"), X=x_train, y=y_train)

# Crear y entrenar modelo inicial
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation="relu", input_shape=(28*28,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# Guardar modelo
model.save(MODEL_PATH)
print("✅ MNIST completo guardado y modelo inicial entrenado.")
