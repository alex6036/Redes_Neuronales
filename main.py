# Script principal del proyecto de redes neuronales
import argparse
from src import phase1_numpy, phase2_compiler, phase3_training, gradio_ui

def main():
    parser = argparse.ArgumentParser(description="El Arquitecto de Redes Neuronales")
    parser.add_argument("--fase", choices=["fase1", "fase2", "fase3", "ui"], default="fase1", help="Fase del proyecto a ejecutar")
    args = parser.parse_args()

    if args.fase == "fase1":
        phase1_numpy.run_example()
    elif args.fase == "fase2":
        phase2_compiler.run_example()
    elif args.fase == "fase3":
        phase3_training.run_example()
    elif args.fase == "ui":
        gradio_ui.run_ui()

if __name__ == "__main__":
    main()

