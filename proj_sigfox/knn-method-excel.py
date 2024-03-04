import pandas as pd
import numpy as np
import time
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, median_absolute_error
from sklearn.metrics.pairwise import haversine_distances
from math import radians
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('sigfox_dataset_rural (1).csv')

X = dataset.iloc[:, :137]
y = dataset[['Latitude', 'Longitude']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


X_train_distance = []
for i in range(len(X_train)):
    print(f'\r{i}/{len(X_train)}',end='')
    all_distances = np.sqrt(np.sum(np.abs(X_train - X_train[i]) ** 2, axis=1))
    X_train_distance.append(all_distances)
X_train_distance = np.array(X_train_distance)

df_distance = pd.DataFrame(X_train_distance)
df_distance.to_csv('X_train_distance.csv', index=False)
