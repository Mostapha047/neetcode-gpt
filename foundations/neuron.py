import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(
        self,
        x: NDArray[np.float64],
        w: NDArray[np.float64],
        b: float,
        activation: str
    ) -> float:

        # convert to numpy arrays
        x = np.array(x)
        w = np.array(w)

        # compute z
        z = np.dot(x, w) + b

        # activation function
        if activation == "sigmoid":
            y = 1 / (1 + np.exp(-z))

        elif activation == "relu":
            y = max(0, z)

        return round(float(y), 5)