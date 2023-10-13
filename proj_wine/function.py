
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

def knn_dt(x, y, number_class):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    model = KNeighborsClassifier(n_neighbors=6)
    model = model.fit(X_train, y_train)

    y_hat = model.predict(X_test)
    acc_knn = accuracy_score(y_test, y_hat)

    model = tree.DecisionTreeClassifier()
    model = model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    acc_Dt = accuracy_score(y_test, y_hat)

    return acc_knn, acc_Dt
