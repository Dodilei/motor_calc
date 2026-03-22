import pandas as pd
from sklearn.model_selection import train_test_split
from surrogate.evaluation import PRSEvaluator
from surrogate.prs import PRSSurrogate

import numpy as np

TRAIN_DATA_FILE = ".data/prop_perfmap.csv"
INPUT_COLS = ["Diameter", "Pitch", "RPM", "V"]
OUTPUT_COLS = ["Thrust", "Torque"]
MODEL_PATH = ".models/prs_propeller_model"


def main():
    # 1. Load Dataset
    train_df = pd.read_csv(TRAIN_DATA_FILE).loc[:, INPUT_COLS + OUTPUT_COLS]
    train_df.dropna(inplace=True)

    X = train_df[INPUT_COLS].values
    y = train_df[OUTPUT_COLS].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.075, random_state=42
    )

    # 2. Initialize Surrogate Model
    degree = 3
    prs_model = PRSSurrogate(degree=degree)

    # 3. Train Model
    prs_model.train(X_train, y_train)
    prs_model.train_error_surrogate(X_train, y_train)

    # 4. Evaluate Performance
    evaluator = PRSEvaluator(cv_splits=5)
    metrics = evaluator.evaluate(prs_model.model, X, y)

    print("--- Robust Validation Metrics ---")
    for target, scores in metrics.items():
        print(f"Target: {OUTPUT_COLS[target].capitalize()}")
        print(f"  Predictive Q² (CV) : {scores['Q2_Predictive']:.4f}")
        print(f"  NRMSE              : {scores['NRMSE'] * 100:.2f}%")
        print(f"  Max Absolute Error : {scores['Max_Absolute_Error']:.4f}")
        print(f"  PRESS Statistic    : {scores['PRESS']:.4e}")
        print("-" * 30)

    # 5. Save the Model Robustly
    prs_model.save(MODEL_PATH)
    print(f"\nModel successfully saved to: {MODEL_PATH}")

    # 6. Verify Loading (Optional)
    loaded_model = PRSSurrogate.load(MODEL_PATH)
    print("\nVerification - Prediction from loaded model:")

    val, err = loaded_model.predict_with_trust(np.array([[18, 8, 5500, 1]]))

    print(f"Predicted Thrust: {val[0][0]:.2f} N")
    print(f"Local Uncertainty (Max Error Predictor): {err[0][0]:.2f} N")


if __name__ == "__main__":
    main()
