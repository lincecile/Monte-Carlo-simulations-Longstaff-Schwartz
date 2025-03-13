from sklearn.linear_model import LinearRegression
import numpy as np

class RegressionEstimator:
    def __init__(self, X, Y, degree=2):
        self.degree = degree
        self.X_poly = self._transform_features(X)
        self.model = LinearRegression()
        self.model.fit(self.X_poly, Y)
        self.coefficients = self.model.coef_
        self.intercept = self.model.intercept_
    
    def _transform_features(self, X):
        return np.column_stack([X**i for i in range(1, self.degree + 1)])
    
    def get_estimator(self, X):
        X_poly = self._transform_features(X)
        return self.model.predict(X_poly)