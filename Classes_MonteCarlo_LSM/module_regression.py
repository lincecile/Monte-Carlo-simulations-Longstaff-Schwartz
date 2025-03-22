from sklearn.linear_model import LinearRegression
import numpy as np

class RegressionEstimator:
    def __init__(self, X, Y, degree=2):
        self.degree = degree
        self.X_poly = self._transform_features(X)
        self.model = LinearRegression()
        self.model.fit(self.X_poly, Y)
        self.coefficients = self.model.coef_.round(3)
        self.intercept = self.model.intercept_.round(3)
    
    def _transform_features(self, X):
        return np.column_stack([X**i for i in range(1, self.degree + 1)])
    
    def get_estimator(self, X):
        X_poly = self._transform_features(X)
        prediction = self.intercept + np.dot(X_poly, self.coefficients)
        print('coeff regression', self.intercept, self.coefficients)
        return self.model.predict(X_poly)
        # return prediction