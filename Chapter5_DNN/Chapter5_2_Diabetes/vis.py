import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes


if __name__ == "__main__":
    dataset = load_diabetes()
    x = dataset.data
    y = dataset.target

    print(f"Feature names:\n{dataset.feature_names}")  # Name der Eigenschafen
    print(f"DESCR:\n{dataset.DESCR}")  # Description

    df = pd.DataFrame(x, columns=dataset.feature_names)
    df["y"] = y
