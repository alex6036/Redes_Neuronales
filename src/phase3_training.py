import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from src.phase2_compiler import compile_model

def load_data():
    """Carga y prepara MNIST."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalización (0-255 -> 0-1)
    x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0
    
    # One-hot encoding de etiquetas
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    return (x_train, y_train), (x_test, y_test)

def train_and_evaluate():
    """Entrena y evalúa un modelo compilado en MNIST."""
    # 1. Cargar datos
    (x_train, y_train), (x_test, y_test) = load_data()

    # 2. Definir arquitectura con nuestro mini-lenguaje
    architecture = "Dense(256, relu) -> Dense(128, relu) -> Dense(10, softmax)"
    model = compile_model(architecture, input_dim=784)

    # 3. Compilar modelo Keras
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # 4. Entrenar
    print("🚀 Entrenando modelo en MNIST...")
    model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

    # 5. Evaluar
    print("\n📊 Evaluando en test set...")
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"✅ Precisión final en test: {acc:.4f}")

def run_example():
    train_and_evaluate()
