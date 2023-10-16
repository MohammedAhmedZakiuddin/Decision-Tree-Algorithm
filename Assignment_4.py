# Mohammed Ahmed Zakiuddin
# 1001675091
# To Run: python Assignment_4.py

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.impute import SimpleImputer

# Calculate the entropy of a list of classes
def entropy(y):
    hist = np.bincount(y) # Count the number of times each value appears
    ps = hist / len(y) # Divide by the total number of values
    return -np.sum([p * np.log2(p) for p in ps if p > 0]) # Sum the entropy of each class

# Represents a node in the decision tree
class Node:
    def __init__(
        self, feature=None, threshold=None, left=None, right=None, *, value=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

# Represents a decision tree
class DecisionTree:
    # Intialize the decision tree
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    # Trains the model using the given training data
    def fit(self, X, y):
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    # Predicts the class for each row in X
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    # Traverses the tree to find the predicted class for a single row. Recursively calls itself
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stopping criteria
        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        # Greedily select the best split according to information gain
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)

        # Print the current node
        feature_name = f"X_{best_feat}"
        condition = f"<={best_thresh:.2f}"
        if depth == 0:
            print(f"Root")
        else:
            print(f"{depth - 1} ", end="")
            for _ in range(depth - 1):
                print("|   ", end="")
            print(f"|--- {feature_name} {condition}")

        # Grow the children that result from the split
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    # Finds the best feature and threshold to use for a split
    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    # Calculates the information gain from a split
    def _information_gain(self, y, X_column, split_thresh):
        # parent loss
        parent_entropy = entropy(y)

        # Generate split
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # Compute the weighted avg. of the loss for the children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # Information gain is difference in loss before vs. after split
        ig = parent_entropy - child_entropy
        return ig

    # Splits the data on a feature that maximizes information gain
    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    # Traverses the tree to find the predicted class for a single row. Recursively calls itself
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    # Finds the most common label in a set of labels
    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
    

if __name__ == "__main__":

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    data = pd.read_csv('btrain.csv')
    data = data.replace('?', np.nan)

    # Fill missing values with 0
    data = data.fillna(0)

    # Convert all columns to strings
    td = data.astype(str)

    X = td.iloc[:, :-1]
    y = td.iloc[:, -1]
    X = np.array(X.values, dtype='float')
    y = np.array(y.values, dtype='int64')

    clf = DecisionTree(max_depth=10)
    clf.fit(X, y)

    vd = pd.read_csv('bvalidate.csv')
    # Preprocess the data to replace NaT values with np.nan
    vd = vd.replace('?', np.nan)

    # Fill missing values with 0
    vd = vd.fillna(0)

    # Convert all columns to strings
    vd = vd.astype(str)

    # Select the features and target variables
    X_1 = vd.iloc[:, :-1]
    y_1 = vd.iloc[:, -1]
    X_1 = np.array(X_1.values, dtype='float')
    y_1 = np.array(y_1.values, dtype='int64')

    # Predict the target variable on the test data
    y_pred = clf.predict(X_1)
    acc = accuracy(y_1, y_pred)

    print("Accuracy:", acc)

    # Load and preprocess the test dataset
    td_test = pd.read_csv('btest.csv')
    td_test = td_test.replace('?', np.nan)
    td_test = td_test.fillna(0)
    td_test = td_test.astype(str)
    X_test = td_test.iloc[:, :-1]
    X_test = np.array(X_test.values, dtype='float')

    # Predict the class labels of the test dataset
    y_pred_test = clf.predict(X_test)

    # Save the predicted class labels to a file
    td_test['class'] = y_pred_test
    td_test.to_csv('btest_predicted.csv', index=False)