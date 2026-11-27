import pickle
import numpy as np

class KNN:
    def __init__(self, k=5, metric="euclidean"):
        if metric not in ("euclidean", "manhattan"):
            raise ValueError("metric must be 'euclidean' or 'manhattan'")
        self.k = k
        self.metric = metric
        self.classes_ = None
        self.X_train_ = None
        self.y_train_ = None

    def fit(self, X, y):
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y)
        self.classes_ = np.unique(self.y_train_)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        distances = self.distance_matrix(X)
        nearest = np.argsort(distances, axis=1, kind="stable")[:, :self.k]
        nearest_labels = self.y_train_[nearest]
        votes = np.stack(
            [(nearest_labels == label).sum(axis=1) for label in self.classes_],
            axis=1,
        )
        return votes / self.k

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as file:
            return pickle.load(file)

    def distance_matrix(self, X):
        if self.metric == "euclidean":
            squared = (np.sum(X ** 2, axis=1)[:, None]
                       + np.sum(self.X_train_ ** 2, axis=1)[None, :]
                       - 2 * X @ self.X_train_.T)
            return np.sqrt(np.maximum(squared, 0))
        distances = np.zeros((X.shape[0], self.X_train_.shape[0]))
        for feature in range(X.shape[1]):
            distances += np.abs(X[:, feature][:, None] - self.X_train_[:, feature][None, :])
        return distances
