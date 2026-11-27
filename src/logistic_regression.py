import numpy as np

class LogisticRegressionOVR:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape

        self.weights = np.zeros((len(self.classes_), n_features))
        self.bias = np.zeros(len(self.classes_))
        self.loss_history = []

        for epoch in range(self.epochs):
            for i, cls in enumerate(self.classes_):
                y_binary = (y == cls).astype(float)

                z = X @ self.weights[i] + self.bias[i]
                predictions = self.sigmoid(z)

                error = predictions - y_binary
                dw = X.T @ error / n_samples
                db = np.mean(error)

                self.weights[i] -= self.learning_rate * dw
                self.bias[i] -= self.learning_rate * db

            # Calculate loss after updating weights
            scores = self.sigmoid(X @ self.weights.T + self.bias)
            y_binary_all = (y[:, None] == self.classes_).astype(float)
            scores = np.clip(scores, 1e-15, 1 - 1e-15)

            loss = -np.mean(
                y_binary_all * np.log(scores)
                + (1 - y_binary_all) * np.log(1 - scores)
            )
            self.loss_history.append(loss)

        return self

    def predict_proba(self, X):
        scores = self.sigmoid(X @ self.weights.T + self.bias)
        return scores / np.maximum(
            scores.sum(axis=1, keepdims=True), 1e-15
        )

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def save(self, filename):
        import pickle
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        import pickle
        with open(filename, "rb") as f:
            return pickle.load(f)
