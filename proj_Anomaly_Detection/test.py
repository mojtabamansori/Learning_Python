import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load data
data_train = pd.read_csv("Herring_TRAIN.tsv", sep='\t')
data_test = pd.read_csv("Herring_TEST.tsv", sep='\t')

X_train = data_train.iloc[:, 1:].values
y_train = data_train.iloc[:, 0].values

X_test = data_test.iloc[:, 1:].values
y_test = data_test.iloc[:, 0].values

X_train_diff = X_train - np.mean(X_train, axis=0)
X_test_diff = X_test - np.mean(X_test ,axis=0)

X_train_var = np.var(X_train, axis=1)
X_test_var = np.var(X_train, axis=1)

X_train = np.hstack((X_train_diff, X_train_var.reshape(-1, 1)))
X_test = np.hstack((X_test_diff, X_test_var.reshape(-1, 1)))

# # Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Particle Swarm Optimization (PSO) parameters
n_particles = 2
n_iterations = 5
n_neighbors_range = range(1, 10)  # Vary the number of neighbors

# Initialize particle positions (k values)
particle_positions = np.random.choice(n_neighbors_range, size=n_particles)

# Initialize global best solution
global_best_accuracy = 0.0
global_best_k = None

# PSO optimization
for iteration in range(n_iterations):
    print(iteration)
    # Evaluate fitness for each particle
    for i, k in enumerate(particle_positions):
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance', algorithm='auto')
        scores = cross_val_score(knn, X_train_scaled, y_train, cv=2)
        accuracy = np.mean(scores)

        # Update personal best solution
        if accuracy > global_best_accuracy:
            global_best_accuracy = accuracy
            global_best_k = k

        # Update particle position (velocity not considered in this simplified example)
        particle_positions[i] = np.random.choice(n_neighbors_range)

# Train k-NN with the best k value
final_knn = KNeighborsClassifier(n_neighbors=global_best_k)
final_knn.fit(X_train_scaled, y_train)

# Predict on test data
y_pred_final = final_knn.predict(X_test_scaled)

# Calculate final accuracy
final_accuracy = accuracy_score(y_test, y_pred_final)
print(f'Final accuracy using PSO-optimized k-NN: {final_accuracy:.2f}')
