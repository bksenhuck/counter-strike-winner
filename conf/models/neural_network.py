"""MLP (Multi-Layer Perceptron) classifier configuration via sklearn.

All hyperparameters can be overridden via environment variables.
Uses sklearn's MLPClassifier — no external deep learning dependency required.
"""

import ast
import os

from sklearn.neural_network import MLPClassifier

PARAMS: dict = {
    "hidden_layer_sizes": ast.literal_eval(
        os.getenv("MLP_HIDDEN_LAYERS", "(128, 64, 32)")
    ),
    "activation": os.getenv("MLP_ACTIVATION", "relu"),
    "max_iter": int(os.getenv("MLP_MAX_ITER", "300")),
    "early_stopping": True,
    "validation_fraction": 0.1,
}


def build_classifier(random_state: int) -> MLPClassifier:
    return MLPClassifier(**PARAMS, random_state=random_state)
