import pandas as pd # Excel für Python
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt


dataset = load_boston()
x = dataset.data
y = dataset.target

df = pd.DataFrame(x, columns=dataset.feature_names)
df["y"] = y

print(df.head(n=20))
print(df.info())
print(df.describe())

df.hist(bins=30, figsize=(15, 15))
plt.show()
