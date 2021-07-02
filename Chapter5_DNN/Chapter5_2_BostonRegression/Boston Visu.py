import pandas as pd # Excel für Python
from sklearn.datasets import load_boston


dataset = load_boston()
x = dataset.data
y = dataset.target

df = pd.DataFrame(x, columns=dataset.feature_names)
df["y"] = y

print(df.head(n=20))
