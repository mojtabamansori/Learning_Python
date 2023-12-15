import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

forest_fires_path = 'forest_fires.csv'
Algerian_forest_fires_path = 'Algerian_forest_fires_dataset.csv'
forest_fires = pd.read_csv(forest_fires_path)
Algerian_forest_fires = pd.read_csv(Algerian_forest_fires_path)
last_column_name = 'Classes'

forest_fires = np.array(forest_fires)

plt.figure(figsize=(10, 7))
plt.boxplot(forest_fires[:,0])
plt.savefig(f'boxplot/his_c_.pdf')
