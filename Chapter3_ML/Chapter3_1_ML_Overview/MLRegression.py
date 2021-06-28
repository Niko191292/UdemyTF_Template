import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
import tf_utils.dummyData as dD
import numpy as np


# Lineare Funktion selber anpassen um den Datensatz zu beschreiben
def model(x):
    m = -6.0 # slope
    b = 12.0 # intercept

    return m * x + b


if __name__ == "__main__":
    # x, y = dD.regression_data()
    x, y = dD.classification_data()
    y_pred = model(x)
    colors = np.array(["red", "blue"])
    plt.scatter(x[:, 0], x[:, 1], color=colors[y])
    # plt.scatter(x, y, marker=5, linewidths=1)
    plt.plot(x, y_pred)
    plt.show()
