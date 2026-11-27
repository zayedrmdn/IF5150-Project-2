import numpy as np

class NeuralNetwork:
    def __init__(self, hidden_layers=(32,), activation="relu",
                 learning_rate=0.01, epochs=200,
                 batch_size=32, random_state=42):
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

        if activation not in ("relu", "tanh"):
            raise ValueError("Activation must be relu or tanh")

    def activate(self, z):
        if self.activation == "relu":
            return np.maximum(0, z)
        return np.tanh(z)

    def activation_derivative(self, z):
        if self.activation == "relu":
            return (z > 0).astype(float)
        return 1 - np.tanh(z) ** 2

    def softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def initialize(self, n_features, n_classes):
        rng = np.random.default_rng(self.random_state)
        sizes = [n_features] + list(self.hidden_layers) + [n_classes]

        self.weights = []
        self.biases = []

        for i in range(len(sizes) - 1):
            scale = np.sqrt(2 / sizes[i])
            self.weights.append(
                rng.normal(0, scale, (sizes[i], sizes[i + 1]))
            )
            self.biases.append(np.zeros((1, sizes[i + 1])))

    def forward(self, X):
        activations = [X]
        z_values = []

        for i in range(len(self.weights) - 1):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            z_values.append(z)
            activations.append(self.activate(z))

        z = activations[-1] @ self.weights[-1] + self.biases[-1]
        z_values.append(z)
        activations.append(self.softmax(z))

        return activations, z_values

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.initialize(X.shape[1], len(self.classes_))

        y_index = np.searchsorted(self.classes_, y)
        y_onehot = np.eye(len(self.classes_))[y_index]

        rng = np.random.default_rng(self.random_state)
        self.loss_history = []

        for epoch in range(self.epochs):
            indices = rng.permutation(len(X))

            for start in range(0, len(X), self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                X_batch = X[batch_idx]
                y_batch = y_onehot[batch_idx]

                activations, z_values = self.forward(X_batch)

                delta = (activations[-1] - y_batch) / len(X_batch)

                grad_w = [None] * len(self.weights)
                grad_b = [None] * len(self.biases)

                for i in reversed(range(len(self.weights))):
                    grad_w[i] = activations[i].T @ delta
                    grad_b[i] = np.sum(
                        delta, axis=0, keepdims=True
                    )

                    if i > 0:
                        delta = (
                            delta @ self.weights[i].T
                        ) * self.activation_derivative(z_values[i - 1])

                for i in range(len(self.weights)):
                    self.weights[i] -= self.learning_rate * grad_w[i]
                    self.biases[i] -= self.learning_rate * grad_b[i]

            probabilities = self.predict_proba(X)
            probabilities = np.clip(probabilities, 1e-15, 1.0)

            loss = -np.mean(
                np.sum(y_onehot * np.log(probabilities), axis=1)
            )
            self.loss_history.append(loss)

        return self

    def predict_proba(self, X):
        activations, _ = self.forward(np.asarray(X, dtype=float))
        return activations[-1]

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
