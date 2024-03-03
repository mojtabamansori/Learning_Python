import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

forest_fires_path = 'forest_fires.csv'
Algerian_forest_fires_path = 'Algerian_forest_fires_dataset.csv'
forest_fires = pd.read_csv(forest_fires_path)
Algerian_forest_fires = pd.read_csv(Algerian_forest_fires_path)

# Drop the 'year' column
forest_fires = forest_fires.drop('year', axis=1)

correlation_matrix = forest_fires.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
