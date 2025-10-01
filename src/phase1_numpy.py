# Fase 1: Implementación con NumPy
import numpy as np

# ------------------------
# Funciones de activación
# ------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

# ------------------------
# Clase Layer
# ------------------------
class Layer:
    def __init__(self, input_size, output_size, activation=None):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        self.activation = activation

    def forward(self, X):
        self.z = np.dot(X, self.weights) + self.bias
        if self.activation:
            self.a = self.activation(self.z)
        else:
            self.a = self.z
        return self.a

# ------------------------
# Clase MLP
# ------------------------
class MLP:
    def __init__(self, layers):
        self.layers = layers

    def predict(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

# ------------------------
# Ejemplo de uso
# ------------------------
def run_example():
    X = np.array([[0.1, 0.5, 0.2, 0.9],
                  [0.3, 0.7, 0.8, 0.2],
                  [0.9, 0.1, 0.4, 0.6]])

    mlp = MLP([
        Layer(input_size=4, output_size=5, activation=relu),
        Layer(input_size=5, output_size=2, activation=sigmoid)
    ])

    y_pred = mlp.predict(X)
    print("Predicciones (Fase 1 - NumPy):\n", y_pred)
