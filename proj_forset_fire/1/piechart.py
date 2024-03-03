import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

forest_fires_path = 'forest_fires.csv'
Algerian_forest_fires_path = 'Algerian_forest_fires_dataset.csv'
forest_fires = pd.read_csv(forest_fires_path)
Algerian_forest_fires = pd.read_csv(Algerian_forest_fires_path)
last_column_name = 'Classes'
last_column_name = forest_fires[last_column_name].astype('category')
i2 = 0
i3 = 0
for i in last_column_name:
    if i == 'fire':
        i2 = i2 + 1
    if i == 'not fire':
        i3 = i3 + 1


y = np.array([i2, i3])
mylabels = ["fire", "not fire"]
myexplode = [0.1, 0.1]
plt.figure(figsize=(6, 4))
plt.pie(y, labels = mylabels, explode = myexplode)
plt.savefig(f'pichart.pdf')
