# Proyecto de Redes Neuronales

Este repositorio contiene la estructura base para el desarrollo de proyectos de redes neuronales, desde la experimentación inicial hasta la implementación y evaluación de modelos.

## Estructura del Proyecto

```
data/                # Datos (crudos, procesados o de ejemplo como MNIST)
models/              # Modelos entrenados (.h5, .pt, etc.)
notebooks/           # Jupyter notebooks de experimentación
src/                 # Código fuente del proyecto
    __init__.py
    phase1_numpy.py   # Fase 1: Implementación con NumPy
    preprocessing.py  # Funciones de preprocesamiento
    model.py          # Definición de la red neuronal
    train.py          # Entrenamiento del modelo
    evaluate.py       # Evaluación del modelo
    predict.py        # Predicciones nuevas
    utils.py          # Funciones auxiliares
tests/               # Pruebas unitarias
requirements.txt     # Librerías necesarias
README.md            # Documentación inicial
.gitignore           # Ignorar cachés, modelos grandes, etc.
main.py              # Script principal
```

## ¿Cómo empezar?

1. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. Coloca tus datos en la carpeta `data/`.
3. Usa los notebooks para experimentar y prototipar modelos.
4. El código fuente está en `src/` y el script principal es `main.py`.

## Ejecución de las fases desde main.py

Para ejecutar cada fase del proyecto, utiliza el archivo principal `main.py`. Por ejemplo:

```bash
python main.py fase1
python main.py fase2
python main.py fase3
```

Asegúrate de que el archivo `main.py` esté preparado para recibir el nombre de la fase como argumento y ejecutar el código correspondiente de la carpeta `src/`.

Ejemplo de uso en `main.py`:
```python
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Por favor, indica la fase a ejecutar: fase1, fase2, fase3...")
    else:
        fase = sys.argv[1]
        if fase == "fase1":
            import src.phase1_numpy
        elif fase == "fase2":
            import src.phase2
        elif fase == "fase3":
            import src.phase3
        else:
            print(f"Fase '{fase}' no reconocida.")
```

## Descripción de carpetas y archivos
- **data/**: Almacena los datos utilizados para entrenar y evaluar los modelos.
- **models/**: Guarda los modelos entrenados en diferentes formatos.
- **notebooks/**: Espacio para experimentación interactiva con Jupyter Notebooks.
- **src/**: Contiene el código principal del proyecto, organizado por funcionalidad.
- **tests/**: Pruebas unitarias para asegurar la calidad del código.
- **requirements.txt**: Lista de librerías necesarias para el proyecto.
- **main.py**: Punto de entrada principal para ejecutar el proyecto.

## Requisitos
- Python 3.8+
- Las librerías listadas en `requirements.txt`

## Contribuciones
Las contribuciones son bienvenidas. Por favor, abre un issue o un pull request para sugerencias o mejoras.

## Licencia
Este proyecto se distribuye bajo la licencia MIT.