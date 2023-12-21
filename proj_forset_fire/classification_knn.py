import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

forest_fires_path = 'forest_fires.csv'
forest_fires = pd.read_csv(forest_fires_path)

forest_fires['target'] = np.where(forest_fires['Classes'] == 'fire', 1, 0)
forest_fires = forest_fires.drop(['Classes'], axis=1)

X = forest_fires.drop(['target'], axis=1)
y = forest_fires['target']

stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

knn_classifier = KNeighborsClassifier()

param_grid = {
    'n_neighbors': [1, 2, 3, 5, 7, 9, 11, 13, 15, 18, 25],
    'weights': ['uniform', 'distance'],
    'p': [1, 2, 3, 4, 5]
}

grid_search = GridSearchCV(knn_classifier, param_grid, scoring='accuracy', cv=stratified_kfold)
grid_search.fit(X, y)

best_knn_classifier = grid_search.best_estimator_

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_knn_classifier.fit(X_train, y_train)

y_pred = best_knn_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))

results = pd.DataFrame(grid_search.cv_results_)

param_columns = ['param_' + param for param in param_grid.keys()]
columns = ['mean_test_score'] + param_columns
plt.figure(figsize=(12, 6))
for i, params in enumerate(param_columns):
    plt.subplot(1, len(param_columns), i + 1)
    grouped = results.groupby(params).mean()['mean_test_score']
    grouped.plot(marker='o')
    plt.title(f'Grid Search CV Results ({params[6:]})')
    plt.xlabel(params[6:])
    plt.ylabel('Mean Test Score')

plt.tight_layout()
plt.show()
