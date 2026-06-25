import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(
        self,
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64]
    ) -> float:

        epsilon = 1e-7

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        n = len(y_true)

        loss = -(1 / n) * np.sum(
            y_true * np.log(y_pred) +
            (1 - y_true) * np.log(1 - y_pred)
        )

        return round(loss, 4)

    def categorical_cross_entropy(
        self,
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64]
    ) -> float:

        epsilon = 1e-7

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        n = len(y_true)

        loss = -(1 / n) * np.sum(
            y_true * np.log(y_pred)
        )

        return round(loss, 4)