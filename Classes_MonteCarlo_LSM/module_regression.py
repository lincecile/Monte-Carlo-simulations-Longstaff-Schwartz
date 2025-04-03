from sklearn.linear_model import LinearRegression
from numpy.polynomial import Polynomial, Laguerre, Hermite, Legendre, Chebyshev
import numpy as np

class RegressionEstimator:
    def __init__(self, X, Y, degree=2, model_type="polynomial"):
        self.X = X
        self.Y = Y
        self.degree = degree
        self.model_type = model_type

    def Regression(self):
        if self.model_type == "polynomial":
            coeffs = np.polynomial.polynomial.polyfit(self.X, self.Y, self.degree)
            linreg = Polynomial(coeffs)
            return linreg(self.X)  
        
        elif self.model_type == "laguerre":
            coeffs = np.polynomial.laguerre.lagfit(self.X, self.Y, self.degree)
            linreg = Laguerre(coeffs)
            return linreg(self.X)  

        elif self.model_type == "hermite":
            coeffs = np.polynomial.hermite.hermfit(self.X, self.Y, self.degree)
            linreg = Hermite(coeffs)
            return linreg(self.X)
        
        elif self.model_type == "legendre":
            coeffs = np.polynomial.legendre.legfit(self.X, self.Y, self.degree)
            linreg = Legendre(coeffs)
            return linreg(self.X)  
        
        elif self.model_type == "chebyshev":
            coeffs = np.polynomial.chebyshev.chebfit(self.X, self.Y, self.degree)
            linreg = Chebyshev(coeffs)
            return linreg(self.X)  

        elif self.model_type == "linear":
            coeffs = np.polyfit(self.X, self.Y, 1)
            return np.polyval(coeffs, self.X)
        
        elif self.model_type == "logarithmic":
            log_X = np.log(self.X)
            coeffs = np.polyfit(log_X, self.Y, 1)
            return np.polyval(coeffs, log_X)

        elif self.model_type == "exponential":
            log_Y = np.log(self.Y)
            coeffs = np.polyfit(self.X, log_Y, 1)
            return np.exp(np.polyval(coeffs, self.X))

        else:
            raise ValueError(f"Modèle '{self.model_type}' non reconnu.")
