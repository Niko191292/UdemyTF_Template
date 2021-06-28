import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from tf_utils.dummyData import regression_data


def mae(y_true: np.ndarray, y_pred: np.ndarray):
    return np.mean(np.abs(y_true - y_pred))


def mse(y_true: np.ndarray, y_pred: np.ndarray):
    return np.mean(np.square(y_true - y_pred))


if __name__ == "__main__":
    x, y = regression_data()
    x = x.reshape(-1, 1)
    print(x.shape)

# Aufteilen von Train und Test Data, disjunktes aufteilen der Daten
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    regr = LinearRegression()
    regr.fit(x_train, y_train)
    score = regr.score(x_test, y_test)
    y_pred = regr.predict(x_test)

    maeSc = mae(y_test, y_pred)
    mseSc = mse(y_test, y_pred)

    print(f"R2-Score: {score}")
    print(f"MAE: {maeSc}")
    print(f"MSE: {mseSc}")
    plt.scatter(x, y)
    plt.plot(x_test, y_pred)
    plt.show()
