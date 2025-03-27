from sklearn.linear_model import LinearRegression
import numpy as np

class RegressionEstimator:
    """ def __init__(self, X, Y, degree=2):
        # Ensure X is properly shaped
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        
        # Create polynomial features
        X_poly = np.column_stack([X**i for i in range(1, degree+1)])
        X_poly = np.column_stack([np.ones(X.shape[0]), X_poly])
        
        # Handle potential multicollinearity with Ridge regression
        # or use np.linalg.lstsq with appropriate rcond
        self.coeffs, _, _, _ = np.linalg.lstsq(X_poly, Y, rcond=1e-10)
        
    def get_estimator(self, X, degree=2):
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        X_poly = np.column_stack([X**i for i in range(1, degree+1)])
        X_poly = np.column_stack([np.ones(X.shape[0]), X_poly])
        return np.maximum(0, X_poly @ self.coeffs)  """
    def __init__(self, X, Y, degree=3):
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
        prediction = self.intercept + np.dot(X_poly, self.coefficients)
        print('coeff regression', self.intercept, self.coefficients)
        return self.model.predict(X_poly)
        # return prediction