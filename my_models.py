# my_models.py
import numpy as np

class MyGaussianNaiveBayes:
    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon
        self.class_probabilities = {}
        self.class_means = {}
        self.class_variances = {}

    def fit(self, X, y):
        classes = np.unique(y)
        self.class_probabilities = {c: np.mean(y == c) for c in classes}
        for c in classes:
            X_class = X[y == c]
            self.class_means[c] = np.mean(X_class, axis=0)
            self.class_variances[c] = np.var(X_class, axis=0) + self.epsilon

    def gaussian_pdf(self, x, mean, var):
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-0.5 * ((x - mean) ** 2) / var)

    def predict(self, X):
        predictions = []
        for x in X:
            probs = {}
            for c in self.class_probabilities:
                prob = np.log(self.class_probabilities[c])
                for i in range(len(x)):
                    mean = self.class_means[c][i]
                    var = self.class_variances[c][i]
                    prob += np.log(self.gaussian_pdf(x[i], mean, var))
                probs[c] = prob
            predictions.append(max(probs, key=probs.get))
        return np.array(predictions)
