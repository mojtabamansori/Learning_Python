import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv ("SemgHandGenderCh2_TRAIN.tsv", sep = '\t')

data = np.array(data)
X_train = data[:,1:]
y_train = data[:,0]

data = pd.read_csv ("SemgHandGenderCh2_TEST.tsv", sep = '\t')
data = np.array(data)
X_test = data[:,1:]
y_test = data[:,0]

model = KNeighborsClassifier(n_neighbors=2)
model = model.fit(X_train, y_train)
y_hat = model.predict(X_test)
acc_knn = accuracy_score(y_test, y_hat)
print(acc_knn)