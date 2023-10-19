import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder



df = pd.read_csv('student-mat.csv')
df = np.array(df)

number_fea = df.shape[1]
for i in range(number_fea):
    col_not_cat = df[:, i]
    arg_to_cat = np.unique(col_not_cat)
    if type(df[10, i]) == str:
        col_not_cat = df[:, i]
        arg_to_cat = np.unique(col_not_cat)
        for number_cat, i_2 in enumerate(arg_to_cat):
            for i_3 in range(len( col_not_cat)):
                if i_2 == col_not_cat[i_3]:
                    col_not_cat[i_3] = number_cat
        df[:, i] = col_not_cat
for i in [30,31,32]:
    col_not_cat = df[:, i]
    for i_2 in range(len(df)):
        if col_not_cat[i_2] < 10:
            col_not_cat[i_2] = 0
        if col_not_cat[i_2] >= 10:
            col_not_cat[i_2] = 1
    df[:, i] = col_not_cat

x_data = df[5:, :30]
y_1 = df[5:, 30]
y_2 = df[5:, 31]
y_3 = df[5:, 32]


label_encoder = LabelEncoder()
y_1 = label_encoder.fit_transform(y_1)
y_2 = label_encoder.fit_transform(y_2)
y_3 = label_encoder.fit_transform(y_3)

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(x_data, y_1, test_size=0.33, random_state=42)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(x_data, y_2, test_size=0.33, random_state=42)
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(x_data, y_3, test_size=0.33, random_state=42)


model = tree.DecisionTreeClassifier()
model = model.fit(X_train_1, y_train_1)
y_hat_1 = model.predict(X_test_1)
print(classification_report(y_test_1, y_hat_1))

model = tree.DecisionTreeClassifier()
model = model.fit(X_train_2, y_train_2)
y_hat_2 = model.predict(X_test_2)
print(classification_report(y_test_2, y_hat_2))

model = tree.DecisionTreeClassifier()
model = model.fit(X_train_3, y_train_3)
y_hat_3 = model.predict(X_test_3)
print(classification_report(y_test_3, y_hat_3))
