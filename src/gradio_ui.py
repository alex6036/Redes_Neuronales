import os
import numpy as np
import tensorflow as tf
import gradio as gr
from tensorflow.keras.models import load_model
from src.phase3_training import load_data, train_and_evaluate

MODEL_PATH = "models/mnist_model.h5"
DATA_PATH = "data/training_data"

# Crear carpeta de datos si no existe
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs("models", exist_ok=True)

# -----------------------
# Cargar o crear modelo
# -----------------------
def get_model():
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
    else:
        # Entrenamiento inicial con MNIST
        train_and_evaluate()  # entrenamos y guardamos modelo
        model = load_model(MODEL_PATH)
    return model

model = get_model()

# -----------------------
# Función para predecir
# -----------------------
def predict_image(image):
    """
    Recibe una imagen, la procesa y predice el número.
    """
    img = np.array(image.convert("L").resize((28,28)))  # convertir a gris y tamaño 28x28
    img = img.reshape(1, 28*28).astype("float32") / 255.0
    pred = model.predict(img)
    return int(np.argmax(pred))

# -----------------------
# Función para agregar datos y entrenar
# -----------------------
def add_data_and_train(image, label):
    """
    Guarda la imagen y etiqueta, entrena el modelo y lo guarda.
    """
    # Guardar imagen y etiqueta en memoria (dataset)
    img_array = np.array(image.convert("L").resize((28,28))).reshape(28*28) / 255.0
    label_int = int(label)

    # Guardar datos en archivo .npz
    data_file = os.path.join(DATA_PATH, "dataset.npz")
    if os.path.exists(data_file):
        data = np.load(data_file)
        X = np.vstack([data["X"], img_array])
        y = np.hstack([data["y"], label_int])
    else:
        X = img_array.reshape(1, -1)
        y = np.array([label_int])

    np.savez(data_file, X=X, y=y)

    # Reentrenar modelo con nuevo dataset
    y_onehot = tf.keras.utils.to_categorical(y, num_classes=10)
    model.fit(X, y_onehot, epochs=3, batch_size=8, verbose=0)
    model.save(MODEL_PATH)

    return f"Imagen agregada y modelo reentrenado. Dataset actual: {len(y)} ejemplos"

# -----------------------
# Crear interfaz Gradio
# -----------------------
with gr.Blocks() as demo:
    gr.Markdown("## Clasificador de Dígitos MNIST 🚀")

    with gr.Tab("Predecir Número"):
        img_input = gr.Image(type="pil", label="Sube un dígito")
        pred_output = gr.Label(num_top_classes=1)
        btn_predict = gr.Button("Predecir")
        btn_predict.click(predict_image, inputs=img_input, outputs=pred_output)

    with gr.Tab("Agregar y Entrenar"):
        img_train = gr.Image(type="pil", label="Sube un dígito")
        label_train = gr.Number(label="Etiqueta (0-9)")
        train_output = gr.Textbox()
        btn_train = gr.Button("Agregar y Entrenar")
        btn_train.click(add_data_and_train, inputs=[img_train, label_train], outputs=train_output)

def run_ui():
    demo.launch()
