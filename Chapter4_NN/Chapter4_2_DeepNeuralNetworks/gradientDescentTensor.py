from inspect import Parameter
import tensorflow as tf
from tf_utils import helper


class Model:
    # Konstruktor
    def __init__(self) -> None:
        self.x = tf.Variable(tf.random.uniform([2], -2., 2.))
        self.learning_rate = 0.002
        self.optimizer = tf.optimizers.SGD(learning_rate=self.learning_rate) # Stochastisch Gradient Descent

    # Funktion die zu minimieren ist
    def loss(self) -> Parameter:
        self.current_loss_val = helper.f(self.x[0], self.x[1])
        return self.current_loss_val

    # Funktion zum trainieren des Models
    def fit(self) -> None:
        self.optimizer.minimize(self.loss, self.x) # loss, variable


model = Model()
gradient_steps = []
x_start = model.x.numpy()
epochs = 5000

for it in range(epochs):
    model.fit()
    if it % 100 == 0:
        x = model.x.numpy()
        y = model.current_loss_val.numpy()
        print(f"X = {x}\tY = {y:.3f}\t Epochs = {it}\n")
        gradient_steps.append(x)

helper.plot_rosenbrock(x_start=x_start, gradient_steps=gradient_steps)
