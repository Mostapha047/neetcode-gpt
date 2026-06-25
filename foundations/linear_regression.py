import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(
        self,
        X: NDArray[np.float64],
        weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:

        # X is (n, m), weights is (m,)
        # returns (n,) predictions

        model_prediction = np.matmul(X, weights)

        return np.round(model_prediction, 5)

    def get_error(
        self,
        model_prediction: NDArray[np.float64],
        ground_truth: NDArray[np.float64]
    ) -> float:

        # Mean Squared Error

        mse = np.mean(
            (np.array(model_prediction) - np.array(ground_truth)) ** 2
        )

        return np.round(mse, 5)