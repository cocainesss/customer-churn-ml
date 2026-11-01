from fastapi.testclient import TestClient
from src.api import app
from src.predict import threshold
import pytest

client = TestClient(app)


@pytest.fixture
def customer_data():
    return {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 90.0,
        "TotalCharges": 180.0
    }


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Customer Churn API is running"
    }


@pytest.mark.parametrize(
    "invalid_contract",
    ["Forever", "Weekly", "", "Three years"]
)
def test_invalid_contract(customer_data, invalid_contract):
    customer_data["Contract"] = invalid_contract

    response = client.post("/predict", json=customer_data)

    assert response.status_code == 422


def test_predict(customer_data):
    response = client.post("/predict", json=customer_data)

    assert response.status_code == 200

    result = response.json()

    assert 0 <= result["churn_probability"] <= 1
    assert result["prediction"] in ["Yes", "No"]
    assert result["risk_level"] in ["Low", "Medium", "High"]

    probability = result["churn_probability"]

    if probability >= threshold:
        assert result["prediction"] == "Yes"
    else:
        assert result["prediction"] == "No"

    if probability >= 0.6:
        assert result["risk_level"] == "High"
    elif probability >= 0.3:
        assert result["risk_level"] == "Medium"
    else:
        assert result["risk_level"] == "Low"


@pytest.mark.parametrize(
    "invalid_charge",
    [-1, -100, -9999]
)
def test_invalid_monthly_charges(customer_data, invalid_charge):
    customer_data["MonthlyCharges"] = invalid_charge

    response = client.post("/predict", json=customer_data)

    assert response.status_code == 422


@pytest.mark.parametrize(
    "invalid_total",
    [-1, -100]
)
def test_invalid_total_charges(customer_data, invalid_total):
    customer_data["TotalCharges"] = invalid_total

    response = client.post("/predict", json=customer_data)

    assert response.status_code == 422


def test_missing_field(customer_data):
    customer_data.pop("tenure")

    response = client.post("/predict", json=customer_data)

    assert response.status_code == 422
