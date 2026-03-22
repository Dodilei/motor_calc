import numpy as np
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import root_mean_squared_error, max_error


class PRSEvaluator:
    def __init__(self, cv_splits: int = 5, random_state: int = 42):
        self.cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    def evaluate(self, estimator, X: np.ndarray, y: np.ndarray) -> dict:
        y_pred_cv = cross_val_predict(estimator, X, y, cv=self.cv)

        y_pred_train = estimator.predict(X)

        results = {}

        for idx in range(y.shape[1]):
            y_true = y[:, idx]

            y_cv = y_pred_cv[:, idx] if y_pred_cv.ndim > 1 else y_pred_cv
            y_train = y_pred_train[:, idx] if y_pred_train.ndim > 1 else y_pred_train

            press = np.sum((y_true - y_cv) ** 2)
            sst = np.sum((y_true - np.mean(y_true)) ** 2)
            q2 = 1 - (press / sst) if sst != 0 else np.nan

            rmse = root_mean_squared_error(y_true, y_train)
            y_range = np.max(y_true) - np.min(y_true)
            nrmse = rmse / y_range if y_range != 0 else np.nan

            max_err = max_error(y_true, y_train)

            results[idx] = {
                "Q2_Predictive": q2,
                "NRMSE": nrmse,
                "Max_Absolute_Error": max_err,
                "PRESS": press,
            }

        return results
