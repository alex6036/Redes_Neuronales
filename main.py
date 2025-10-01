# Script principal del proyecto de redes neuronales
import argparse
import os
import shutil
from src import phase1_numpy, phase2_compiler, phase3_training, gradio_ui
from src.phase3_training import train_and_evaluate

def main():
    parser = argparse.ArgumentParser(description="El Arquitecto de Redes Neuronales")
    parser.add_argument(
        "--fase",
        choices=["fase1", "fase2", "fase3", "ui", "reset"],  # <-- agregamos "reset"
        default="fase1",
        help="Fase del proyecto a ejecutar"
    )
    args = parser.parse_args()

    if args.fase == "fase1":
        phase1_numpy.run_example()
    elif args.fase == "fase2":
        phase2_compiler.run_example()
    elif args.fase == "fase3":
        phase3_training.run_example()
    elif args.fase == "ui":
        gradio_ui.run_ui()
    elif args.fase == "reset":
        # -----------------------
        # Borrar dataset
        # -----------------------
        if os.path.exists("data/training_data"):
            shutil.rmtree("data/training_data")
        os.makedirs("data/training_data", exist_ok=True)

        # -----------------------
        # Borrar modelo
        # -----------------------
        if os.path.exists("models/mnist_model.h5"):
            os.remove("models/mnist_model.h5")

        # -----------------------
        # Entrenar nuevo modelo
        # -----------------------
        train_and_evaluate()
        print("✅ Reset completo: dataset y modelo borrados, modelo nuevo entrenado.")

if __name__ == "__main__":
    main()
