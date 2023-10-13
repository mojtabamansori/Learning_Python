import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import classification_report


df = pd.read_csv('dataset/wine-quality-red.csv')
df = np.array(df)

y = df[:, -1]
x = df[:, :-1]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

model = tree.DecisionTreeClassifier()
model = model.fit(X_train, y_train)

y_hat = model.predict(X_test)
print(classification_report(y_test, y_hat))
