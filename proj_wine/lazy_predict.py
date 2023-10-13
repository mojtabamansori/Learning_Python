import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lazypredict



df = pd.read_csv('dataset/wine-quality-red.csv')
df = np.array(df)

y = df[:, -1]
x = df[:, :-1]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
