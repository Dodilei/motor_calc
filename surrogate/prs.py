import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score


class PRSSurrogate:
    def __init__(self, degree=3):
        self.degree = degree
        self.model = self._build_pipeline()

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

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        y_pred = np.array(self.model.predict(X_test))

        # Calculate metrics for each output variable independently
        metrics = {}
        idx: int
        for idx in range(y_test.shape[1]):
            metrics[idx] = {
                "RMSE": root_mean_squared_error(y_test[:, idx], y_pred[:, idx]),
                "R2": r2_score(y_test[:, idx], y_pred[:, idx]),
            }
        return metrics

    def save(self, filepath: str):
        joblib.dump(self.model, filepath)

    @classmethod
    def load(cls, filepath: str):
        instance = cls()
        instance.model = joblib.load(filepath)
        return instance
