from sklearn.linear_model import LinearRegression
import numpy as np

class RegressionEstimator:
    def __init__(self, X, Y, degree=3, poly_type="standard"):
        self.degree = degree
        self.poly_type = poly_type

    def Regression(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        if self.poly_type == "standard":
            linreg = np.polyfit(X, Y, self.degree)
            return np.polyval(linreg, X)
        else:
            raise ValueError(f"Erreur dans type de poly: {self.poly_type}")

    