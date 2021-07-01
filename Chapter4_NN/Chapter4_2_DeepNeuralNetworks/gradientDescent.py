import numpy as np
from tf_utils import helper


# Zufälliger Startpunkt
x0 = np.random.uniform(-2, 2)
x1 = np.random.uniform(-2, 2)
x_start = (x0, x1)
y_start = helper.f(x0, x1)

print("Global minimum: ", 1, 1) # Globales minimum der Rosenbrock FUnktion
print("X-Start = ", x_start)
print("Y-Start = ", y_start)
helper.plot_rosenbrock(x_start)

# eta n -> Intervall 0..1 ohne 0
learning_rate = 0.003 # [0.001 - 0.00001]
# Durchläufe Epochen
num_iterations = 4000

# Koordinaten im laufe der iterationen
gradient_steps = []

for it in range(num_iterations):
    x0 = x0 - learning_rate * helper.f_prime_x0(x0, x1) # New = old - lr * derivative(Ableitung)
    # Substraktion wegen dem Minimum, ABleitung zeigt in richtung der Anstiegs
    x1 = x1 - learning_rate * helper.f_prime_x1(x0, x1)
    y = helper.f(x0, x1)
    if it % 100 == 0:
        print("Epochs = ", it)
        print(f"x0 = {x0:.4}\tx1 = {x1:.4}\ty = {y:.4}\n")
        gradient_steps.append((x0, x1))

x_end = (x0, x1)
y_end = helper.f(x0, x1)
print("X_end", x_end)
print("Y_end", y_end)
helper.plot_rosenbrock(x_start=x_start, gradient_steps=gradient_steps)
