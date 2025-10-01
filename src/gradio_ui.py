import os
import numpy as np
import tensorflow as tf
import gradio as gr
from PIL import Image

MODEL_PATH = "models/mnist_model.h5"
DATA_PATH = "data/training_data"
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs("models", exist_ok=True)

# -----------------------
# Cargar modelo existente
# -----------------------
def get_model():
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print("✅ Modelo cargado desde disco.")
        except:
            raise RuntimeError("⚠️ Error cargando modelo. Entrena un modelo nuevo primero.")
    else:
        raise FileNotFoundError("⚠️ Modelo no encontrado. Ejecuta el script de entrenamiento inicial primero.")
    return model

model = get_model()

# -----------------------
# Función de predicción y reentrenamiento
# -----------------------
def predict_and_correct(image: Image.Image, correct_label: str = None):
    # Preprocesar imagen
    img = image.convert("L").resize((28,28))
    img_array = np.array(img).reshape(1, 28*28).astype("float32") / 255.0

    # Predicción inicial
    pred = model.predict(img_array)
    pred_class = int(np.argmax(pred))

    # Reentrenar si el usuario corrigió la etiqueta
    if correct_label is not None:
        correct_label_int = int(correct_label)

        # Cargar MNIST completo
        mnist_file = "data/mnist_full.npz"
        mnist_data = np.load(mnist_file)
        X_base = mnist_data["X"]
        y_base = mnist_data["y"]

        # Cargar dataset incremental
        data_file = os.path.join(DATA_PATH, "dataset.npz")
        if os.path.exists(data_file):
            new_data = np.load(data_file)
            X_new = new_data["X"]
            y_new = new_data["y"]
            X_combined = np.vstack([X_base, X_new, img_array])
            y_combined = np.vstack([
                y_base,
                tf.keras.utils.to_categorical(y_new, num_classes=10),
                tf.keras.utils.to_categorical([correct_label_int], num_classes=10)
            ])
        else:
            X_combined = np.vstack([X_base, img_array])
            y_combined = np.vstack([y_base,
                                    tf.keras.utils.to_categorical([correct_label_int], num_classes=10)])

        # Guardar dataset incremental
        if os.path.exists(data_file):
            np.savez(data_file,
                     X=np.vstack([X_new, img_array]),
                     y=np.hstack([y_new, correct_label_int]))
        else:
            np.savez(data_file, X=img_array, y=np.array([correct_label_int]))

        # Reentrenar modelo
        dataset = tf.data.Dataset.from_tensor_slices((X_combined, y_combined)).batch(32)
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(dataset, epochs=3, verbose=0)
        model.save(MODEL_PATH)

        pred_class = correct_label_int  # reflejar corrección inmediata

    return pred_class

# -----------------------
# Interfaz Gradio
# -----------------------
with gr.Blocks() as demo:
    gr.Markdown("## Clasificador de Dígitos MNIST 🚀")
    gr.Markdown("Sube un dígito, recibe la predicción y corrige si es necesario.")

    img_input = gr.Image(type="pil", label="Sube un dígito")

    # Dropdown seguro para etiquetas
    label_input = gr.Dropdown(choices=[str(i) for i in range(10)],
                              label="Etiqueta correcta (opcional)")

    output_label = gr.Label(num_top_classes=1, label="Predicción")
    btn = gr.Button("Predecir y Reentrenar si es necesario")
    btn.click(predict_and_correct, inputs=[img_input, label_input], outputs=output_label)

def run_ui():
    demo.launch()
