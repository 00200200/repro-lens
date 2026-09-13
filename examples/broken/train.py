"""Deliberately broken inputs for the static checker; this script is never executed."""

import random

import numpy as np
from sklearn.model_selection import KFold, train_test_split


def train(X, y):
    split = train_test_split(X, y, test_size=0.2)
    cv = KFold(n_splits=5, shuffle=True)
    rng = np.random.default_rng()
    augment_rng = random.Random()
    return split, cv, rng, augment_rng
