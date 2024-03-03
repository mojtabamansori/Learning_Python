import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

forest_fires_path = 'forest_fires.csv'
forest_fires = pd.read_csv(forest_fires_path)

forest_fires['target'] = np.where(forest_fires['Classes'] == 'fire', 1, 0)
forest_fires = forest_fires.drop(['Classes'], axis=1)

X = forest_fires.drop(['target'], axis=1)
y = forest_fires['target']

stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

dt_classifier = DecisionTreeClassifier()

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 15, 17, 20, 23, 25],
    'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9]
}

grid_search = GridSearchCV(dt_classifier, param_grid, scoring='accuracy', cv=stratified_kfold)
grid_search.fit(X, y)

best_dt_classifier = grid_search.best_estimator_

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_dt_classifier.fit(X_train, y_train)

y_pred = best_dt_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))

# Save results to CSV
cv_results_df = pd.DataFrame(grid_search.cv_results_)
cv_results_df.to_csv('cv_results.csv', index=False)

# Plot Grid Search CV Results
param_columns = ['param_' + param for param in param_grid.keys()]
columns = ['mean_test_score'] + param_columns
plt.figure(figsize=(12, 6))
for i, params in enumerate(param_columns):
    plt.subplot(1, len(param_columns), i + 1)
    grouped = cv_results_df.groupby(params).mean()['mean_test_score']
    grouped.plot(marker='o')
    plt.title(f'Grid Search CV Results ({params[6:]})')
    plt.xlabel(params[6:])
    plt.ylabel('Mean Test Score')

plt.tight_layout()
plt.show()

# Plot and save Decision Tree
plt.figure(figsize=(20, 10))
feature_names_list = list(X.columns)
plot_tree(best_dt_classifier, filled=True, feature_names=feature_names_list, class_names=['non-fire', 'fire'])
plt.title('Best Decision Tree Classifier')
plt.savefig('best_decision_tree.png')
plt.show()
