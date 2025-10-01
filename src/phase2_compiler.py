import tensorflow as tf

# Diccionario para mapear strings a funciones de activación
ACTIVATIONS = {
    "relu": "relu",
    "sigmoid": "sigmoid",
    "softmax": "softmax",
    "tanh": "tanh",
    "linear": "linear"
}

def compile_model(architecture_string, input_dim=None):
    """
    Compila un modelo de Keras a partir de una descripción textual.
    
    Parámetros:
    - architecture_string: str con la arquitectura, ej. "Dense(256, relu) -> Dense(128, relu) -> Dense(10, softmax)"
    - input_dim: int, dimensión de entrada (solo necesario para la primera capa)
    
    Devuelve:
    - modelo: instancia de tf.keras.Sequential
    """
    layers_def = architecture_string.split("->")
    model = tf.keras.Sequential()
    
    for i, layer_def in enumerate(layers_def):
        layer_def = layer_def.strip()  # limpiar espacios
        if not layer_def.startswith("Dense"):
            raise ValueError(f"Tipo de capa no soportado: {layer_def}")

        # Parseamos Dense(units, activation)
        params = layer_def[len("Dense("):-1]  # contenido entre paréntesis
        units_str, activation_str = [p.strip() for p in params.split(",")]
        units = int(units_str)
        activation = ACTIVATIONS.get(activation_str.lower(), None)

        # Añadimos la capa
        if i == 0 and input_dim is not None:
            model.add(tf.keras.layers.Dense(units, activation=activation, input_dim=input_dim))
        else:
            model.add(tf.keras.layers.Dense(units, activation=activation))
    
    return model

# ------------------------
# Ejemplo de uso
# ------------------------
def run_example():
    architecture = "Dense(256, relu) -> Dense(128, relu) -> Dense(10, softmax)"
    model = compile_model(architecture, input_dim=784)
    model.summary()
