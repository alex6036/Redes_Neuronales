# Script principal del proyecto de redes neuronales
import argparse
from src import phase1_numpy, phase2_compiler, phase3_training

def main():
    parser = argparse.ArgumentParser(description="El Arquitecto de Redes Neuronales")
    parser.add_argument("--fase", choices=["fase1", "fase2", "fase3"], default="fase1", help="Fase del proyecto a ejecutar")
    args = parser.parse_args()

    if args.fase == "fase1":
        print("🚀 Ejecutando Fase 1: Red Neuronal desde cero con NumPy...\n")
        phase1_numpy.run_example()

    elif args.fase == "fase2":
        print("🛠️ Ejecutando Fase 2: Compilador de Arquitecturas...\n")
        phase2_compiler.run_example()

    elif args.fase == "fase3":
        print("📚 Ejecutando Fase 3: Entrenamiento y Evaluación en MNIST...\n")
        phase3_training.run_example()

if __name__ == "__main__":
    main()
