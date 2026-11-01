from pathlib import Path
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "churn_model_real.joblib"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "churn_threshold_real.joblib"

model = joblib.load(MODEL_PATH)
threshold = float(joblib.load(THRESHOLD_PATH))

positive_class = "Yes"


def predict_churn(customer: pd.DataFrame):
    probabilities = model.predict_proba(customer)

    yes_index = list(model.classes_).index(positive_class)
    churn_probability = float(probabilities[0, yes_index])

    prediction = (
        "Yes" if churn_probability >= threshold else "No"
    )

    if churn_probability >= 0.6:
        risk_level = "High"
    elif churn_probability >= 0.3:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "churn_probability": churn_probability,
        "prediction": prediction,
        "risk_level": risk_level
    }
