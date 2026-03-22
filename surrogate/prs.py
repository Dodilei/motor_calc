import joblib
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression


class PRSSurrogate:
    def __init__(self, degree=3):
        self.degree = degree
        self.model = self._build_pipeline()
        self.error_model = KNeighborsRegressor(n_neighbors=5, weights="distance")
        self.scaler_error = StandardScaler()

    def _build_pipeline(self):
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=self.degree, include_bias=False)),
                ("regressor", LinearRegression()),
            ]
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = self.model.predict(X)
        # Ensure predictions maintain the original output column names if available
        return np.array(predictions)

    def save(self, filepath: str):
        joblib.dump(self.model, filepath + ".prs")
        joblib.dump(self.scaler_error, filepath + ".errscl")
        joblib.dump(self.error_model, filepath + ".errmd")

    @classmethod
    def load(cls, filepath: str):
        instance = cls()
        instance.model = joblib.load(filepath + ".prs")
        instance.error_model = joblib.load(filepath + ".errmd")
        instance.scaler_error = joblib.load(filepath + ".errscl")
        return instance

    def train_error_surrogate(self, X_train: np.ndarray, y_train: np.ndarray):
        """Trains a secondary model to predict the absolute local error."""
        y_pred = self.model.predict(X_train)
        residuals = np.abs(y_train - y_pred)

        X_scaled = self.scaler_error.fit_transform(X_train)
        self.error_model.fit(X_scaled, residuals)

    def predict_with_trust(self, X: np.ndarray):
        """Returns predictions and the estimated local error (uncertainty)."""
        y_pred = self.model.predict(X)

        X_scaled = self.scaler_error.transform(X)
        estimated_error = self.error_model.predict(X_scaled)

        return y_pred, estimated_error
