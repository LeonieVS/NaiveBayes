import numpy as np
import pandas as pd

def fit(X, y):
    prior = 1 / len(np.unique(y))

    matrix = [[feat for feat, label in zip(X, y) if label == clss] for clss in np.unique(y)]
    classprior = []
    for x in matrix:
        loggie = np.log((len(x)/len(X.todense()))
        classprior.append(loggie)
    np.array(classprob)
    featprob = []
    for x in matrix:
        sum something axis=0
    return len(matrix[0]), prior
    #Co-occurrence counts: je berekent de joint probability ipv de conditional prob

def predict(X):
    predicted_labels = []
    for x in X:
        posterior = (prior * likelihood) / evidence

        predicted_labels.append(max(predict))
    return predicted_labels
