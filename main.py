# Script principal del proyecto de redes neuronales
import argparse
from src import phase1_numpy

def main():
    parser = argparse.ArgumentParser(description="El Arquitecto de Redes Neuronales")
    parser.add_argument("fase", choices=["fase1"], help="Fase del proyecto a ejecutar")
    args = parser.parse_args()

    if args.fase == "fase1":
        phase1_numpy.run_example()

if __name__ == "__main__":
    main()
