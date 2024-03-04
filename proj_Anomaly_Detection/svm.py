import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data_train = pd.read_csv("Herring/Herring_TRAIN.tsv", sep='\t')
data_test = pd.read_csv("Herring/Herring_TEST.tsv", sep='\t')

X_train = data_train.iloc[:, 1:].values
y_train = data_train.iloc[:, 0].values
X_test = data_test.iloc[:, 1:].values
y_test = data_test.iloc[:, 0].values


svm_classifier = SVC(kernel='rbf', C=20.0, gamma=0.01, class_weight='balanced')
svm_classifier.fit(X_train, y_train)
y_pred_svm = svm_classifier.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
print(f'Final accuracy using SVM with class weights: {svm_accuracy:.2f}')
