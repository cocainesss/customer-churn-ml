from pathlib import Path
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / 'models' / 'churn_model.joblib'

artifact = joblib.load(MODEL_PATH)

model = artifact['pipeline']
threshold = artifact['threshold']
positive_class = artifact['positive_class']

customer = pd.DataFrame([{
    "gender": "Male",
    "senior_citizen": 0,
    "partner": "Yes",
    "dependents": "No",
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "DSL",
    "online_security": "No",
    "online_backup": "Yes",
    "device_protection": "No",
    "contract": "Month-to-month",
    "monthly_charges": 85.0
}])

def predict_churn(customer):
    probabilities = model.predict_proba(customer)

    yes_index = list(model.classes_).index(positive_class)
    churn_probability = probabilities[0, yes_index]

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
        "churn_probability": float(churn_probability),
        "prediction": prediction,
        "risk_level": risk_level
    }

if __name__ == "__main__":
    print(predict_churn(customer))