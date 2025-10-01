import tensorflow as tf
import numpy as np
import os

# -----------------------
# Paths
# -----------------------
MODEL_PATH = "models/mnist_model.h5"
os.makedirs("models", exist_ok=True)

# -----------------------
# Cargar MNIST
# -----------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizar y aplanar imágenes
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0

# One-hot encoding de etiquetas
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# -----------------------
# Crear modelo
# -----------------------
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation="relu", input_shape=(28*28,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

# -----------------------
# Compilar modelo
# -----------------------
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------
# Entrenar modelo
# -----------------------
print("🚀 Entrenando modelo inicial con MNIST completo...")
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# -----------------------
# Evaluar
# -----------------------
loss, acc = model.evaluate(x_test, y_test)
print(f"✅ Precisión en test set: {acc:.4f}")

# -----------------------
# Guardar modelo
# -----------------------
model.save(MODEL_PATH)
print(f"✅ Modelo guardado en {MODEL_PATH}")
