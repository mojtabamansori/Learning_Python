import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from func_his import his

data = pd.read_excel('New Microsoft Excel Worksheet.xlsx')
class_labels = data['Class']
label_encoder = LabelEncoder()
data['Class'] = label_encoder.fit_transform(class_labels)
data = np.array(data)

y = data[:, -1]
x = data[:, :-1]

# his(x)
i1 = 0
i2 = 0
for i in y:
    if int(i) == 0:
        i1 = i1 + 1
    if int(i) == 1:
        i2 = i2 + 1
y_pie = [i1, i2]
plt.pie(y_pie)
plt.show()
