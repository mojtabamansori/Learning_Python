import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

forest_fires_path = 'forest_fires.csv'
Algerian_forest_fires_path = 'Algerian_forest_fires_dataset.csv'
forest_fires = pd.read_csv(forest_fires_path)
Algerian_forest_fires = pd.read_csv(Algerian_forest_fires_path)
last_column_name = 'Classes'
# تبدیل ستون آخر به دسته‌ای
last_column_name = forest_fires[last_column_name].astype('category')

# ساخت نمودار Scatter Plot با استفاده از hue بر اساس ستون آخر
sns.scatterplot(x=forest_fires['RH'], y=forest_fires['Ws'], hue=last_column_name, palette='viridis')

plt.title('Scatter Plot cor = 0.24')
plt.xlabel('RH')
plt.ylabel('Ws')
plt.show()
