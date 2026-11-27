import pickle

import numpy as np

from id3 import ID3


class RandomForest:
    def __init__(self, n_trees=50, max_features=None, max_depth=None,
                 min_samples_split=2, categorical_columns=None, random_state=None):
        self.n_trees = n_trees
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.categorical_columns = categorical_columns
        self.random_state = random_state
        self.trees_ = []
        self.classes_ = None
        self.class_counts_ = None
        self.features_per_split_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.class_counts_ = self.count_classes(y)
        self.features_per_split_ = self.choose_features_per_split(X.shape[1])
        seeds = self.tree_seeds()
        self.trees_ = []
        for seed in seeds:
            rows = self.bootstrap_rows(len(y), seed)
            tree = ID3(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.features_per_split_,
                categorical_columns=self.categorical_columns,
                random_state=seed,
            )
            tree.fit(X[rows], y[rows])
            self.trees_.append(tree)
        return self

    def predict(self, X):
        X = np.asarray(X)
        votes = np.array([tree.predict(X) for tree in self.trees_])
        return np.array([self.majority_vote(votes[:, row]) for row in range(X.shape[0])])

    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as file:
            return pickle.load(file)

    def choose_features_per_split(self, number_of_features):
        if self.max_features is not None:
            return self.max_features
        return max(1, int(np.sqrt(number_of_features)))

    def tree_seeds(self):
        generator = np.random.RandomState(self.random_state)
        return generator.randint(0, 2 ** 31 - 1, size=self.n_trees)

    def bootstrap_rows(self, number_of_rows, seed):
        generator = np.random.RandomState(seed)
        return generator.randint(0, number_of_rows, size=number_of_rows)

    def count_classes(self, y):
        values, counts = np.unique(y, return_counts=True)
        return {value: int(count) for value, count in zip(values, counts)}

    def majority_vote(self, votes):
        values, counts = np.unique(votes, return_counts=True)
        best = values[counts == counts.max()]
        if len(best) == 1:
            return best[0]
        return max(best, key=lambda label: self.class_counts_.get(label, 0))
