import pandas as pd
import numpy as np
from function import knn_dt

df = pd.read_csv('dataset/wine-quality-red.csv')
df = np.array(df)
y = df[:, -1]
x = df[:, :-1]
acc_knn, acc_Dt = knn_dt(x, y, 6)
print('acc_knn = ', acc_knn, '\nacc_dt = ', acc_Dt)
