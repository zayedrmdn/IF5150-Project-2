import pickle

import numpy as np


def is_missing(value):
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return False


def entropy_of_labels(labels):
    values, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


class Node:
    def __init__(self, prediction):
        self.prediction = prediction
        self.feature = None
        self.kind = None
        self.threshold = None
        self.fill = None
        self.left = None
        self.right = None
        self.branches = None


class ID3:
    def __init__(self, max_depth=None, min_samples_split=2, max_features=None,
                 categorical_columns=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.categorical_columns = categorical_columns
        self.random_state = random_state
        self.classes_ = None
        self.root_ = None
        self.is_categorical_ = None
        self.rng_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.is_categorical_ = np.zeros(X.shape[1], dtype=bool)
        if self.categorical_columns is not None:
            self.is_categorical_[list(self.categorical_columns)] = True
        self.rng_ = np.random.RandomState(self.random_state)
        self.root_ = self.build_tree(X, y, 0)
        return self

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self.predict_row(self.root_, row) for row in X])

    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as file:
            return pickle.load(file)

    def build_tree(self, X, y, depth):
        node = Node(self.most_common_label(y))
        if self.should_stop(y, depth):
            return node
        X_clean = self.impute_missing(X)
        split = self.best_split(X_clean, y)
        if split is None:
            return node
        feature, threshold = split
        node.feature = feature
        node.threshold = threshold
        node.kind = "categorical" if self.is_categorical_[feature] else "numeric"
        node.fill = self.fill_value(X[:, feature], self.is_categorical_[feature])
        if node.kind == "categorical":
            column = X_clean[:, feature]
            node.branches = {}
            for value in np.unique(column):
                rows = column == value
                node.branches[value] = self.build_tree(X[rows], y[rows], depth + 1)
        else:
            column = X_clean[:, feature].astype(float)
            left_rows = column <= threshold
            node.left = self.build_tree(X[left_rows], y[left_rows], depth + 1)
            node.right = self.build_tree(X[~left_rows], y[~left_rows], depth + 1)
        return node

    def predict_row(self, node, row):
        while node.feature is not None:
            value = row[node.feature]
            if is_missing(value):
                value = node.fill
            if node.kind == "categorical":
                next_node = node.branches.get(value)
                if next_node is None:
                    return node.prediction
                node = next_node
            else:
                node = node.left if float(value) <= node.threshold else node.right
        return node.prediction

    def should_stop(self, y, depth):
        if len(np.unique(y)) == 1:
            return True
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        return len(y) < self.min_samples_split

    def most_common_label(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def candidate_features(self, number_of_features):
        if self.max_features is None:
            return np.arange(number_of_features)
        size = min(self.max_features, number_of_features)
        return np.sort(self.rng_.choice(number_of_features, size=size, replace=False))

    def best_split(self, X, y):
        base_entropy = entropy_of_labels(y)
        best_feature = None
        best_threshold = None
        best_gain = 0.0
        for feature in self.candidate_features(X.shape[1]):
            column = X[:, feature]
            if self.is_categorical_[feature]:
                gain = self.categorical_gain(column, y, base_entropy)
                threshold = None
            else:
                gain, threshold = self.numeric_gain(column, y, base_entropy)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
        if best_feature is None:
            return None
        return best_feature, best_threshold

    def split_entropy(self, y, left_rows):
        left = y[left_rows]
        right = y[~left_rows]
        total = len(y)
        left_part = len(left) / total * entropy_of_labels(left)
        right_part = len(right) / total * entropy_of_labels(right)
        return left_part + right_part

    def categorical_gain(self, column, y, base_entropy):
        values = np.unique(column)
        if len(values) < 2:
            return 0.0
        weighted = 0.0
        for value in values:
            rows = column == value
            weighted += len(y[rows]) / len(y) * entropy_of_labels(y[rows])
        return base_entropy - weighted

    def numeric_gain(self, column, y, base_entropy):
        column = column.astype(float)
        values = np.unique(column)
        if len(values) < 2:
            return 0.0, None
        best_gain = 0.0
        best_threshold = None
        for index in range(len(values) - 1):
            threshold = (values[index] + values[index + 1]) / 2.0
            left_rows = column <= threshold
            gain = base_entropy - self.split_entropy(y, left_rows)
            if gain > best_gain:
                best_gain = gain
                best_threshold = float(threshold)
        return best_gain, best_threshold

    def missing_mask(self, column):
        if column.dtype.kind == "f":
            return np.isnan(column)
        return np.array([is_missing(value) for value in column], dtype=bool)

    def fill_value(self, column, categorical):
        missing = self.missing_mask(column)
        kept = column[~missing]
        if len(kept) == 0:
            return "" if categorical else 0.0
        if categorical:
            values, counts = np.unique(kept, return_counts=True)
            return values[np.argmax(counts)]
        return float(np.median(kept.astype(float)))

    def impute_missing(self, X):
        X_clean = None
        for feature in range(X.shape[1]):
            column = X[:, feature]
            missing = self.missing_mask(column)
            if not missing.any():
                continue
            if X_clean is None:
                X_clean = np.array(X, copy=True)
            X_clean[missing, feature] = self.fill_value(column, self.is_categorical_[feature])
        if X_clean is None:
            return X
        return X_clean
