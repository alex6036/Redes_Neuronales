import os
import shutil
from src.phase3_training import train_and_evaluate

# -----------------------
# Borrar dataset antiguo
# -----------------------
DATA_PATH = "data/training_data"
if os.path.exists(DATA_PATH):
    shutil.rmtree(DATA_PATH)
os.makedirs(DATA_PATH, exist_ok=True)

# -----------------------
# Borrar modelo antiguo
# -----------------------
MODEL_PATH = "models/mnist_model.h5"
if os.path.exists(MODEL_PATH):
    os.remove(MODEL_PATH)

# -----------------------
# Entrenar modelo limpio
# -----------------------
train_and_evaluate()
print("✅ Reset completo: dataset y modelo borrados, modelo nuevo entrenado.")
