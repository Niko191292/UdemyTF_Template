from typing import Tuple
import numpy as np
from numpy.core.fromnumeric import size


def get_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """OR-Function dataset"""
    x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])  # Input in das OR-Gatter
    y = np.array([[0], [1], [1], [1]])  # Ergebnis eines OR-Gatters
    return x, y


def accuracyScore(y_true: np.ndarray, y_pred: np.ndarray):
    N = y_true.shape[0] # Anzahl der Datenpunkte
    accuracy = np.sum(y_true == y_pred) / N # Wie oft richtig
    return accuracy


def step_function(input_sig: np.ndarray) -> np.ndarray:
    output_signal = (input_sig > 0.0).astype(np.int_)
    return output_signal


class Perceptron:
    def __init__(self, learning_rate: float, input_dim: int) -> None:
        """Initialisierung des Perceptron Objekts"""
        self.learning_rate = learning_rate
        self.input_dim = input_dim

        # Erstellen der Gewichtsmatrix, zufällig gewählte Gewichte
        self.w = np.random.uniform(-1, 1, size=(self.input_dim, 1))

    def _update_weights(self, x: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> None:
        error = (y - y_pred)
        delta = error * x

        for delta_i in delta:
            self.w = self.w + self.learning_rate * delta_i.reshape(-1, 1)

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int = 1) -> None:
        """Trainieren des Models. 1.Vorhersagte, 2.Gewichtung anpassen, 3.Genauigkeit, 4.Ausgabe"""
        for epoch in range(1, epochs + 1):
            y_pred = self.predict(x)
            self._update_weights(x, y, y_pred)
            accuracy = accuracyScore(y, y_pred)
            print(f"Epoch: {epoch} Accuracy: {accuracy}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        input_sig = np.dot(x, self.w)
        output_sig = step_function(input_sig)
        return output_sig

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> None:
        y_pred = self.predict(x)
        return accuracyScore(y, y_pred)


if __name__ == "__main__":
    x, y = get_dataset()

    input_dim = x.shape[1]  # Anzahl der Features
    learning_rate = 0.5

    p = Perceptron(learning_rate, input_dim)

    p.train(x, y, epochs=10)
