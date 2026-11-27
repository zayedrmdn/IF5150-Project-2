import pickle

import numpy as np

class Ensemble:
    def __init__(self, models):
        # models: a list of model objects that have fit(X, y) and predict(X)
        self.models = models
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        votes = np.array([model.predict(X) for model in self.models])
        counts = np.stack(
            [(votes == label).sum(axis=0) for label in self.classes_],
            axis=1,
        )
        return self.classes_[np.argmax(counts, axis=1)]

    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as file:
            return pickle.load(file)
