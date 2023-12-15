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

def his(x):
    for i in x:
        plt.figure(figsize=(6, 4))
        sns.histplot(x[i])
        plt.savefig(f'histogram/his_c_{i}.pdf')

his(forest_fires)
