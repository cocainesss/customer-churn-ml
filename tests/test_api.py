from fastapi.testclient import TestClient
from src.api import app
from src.predict import threshold
import pytest
client = TestClient(app)

@pytest.fixture
def customer_data():
    return {
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
    customer_data["contract"] = invalid_contract
    response = client.post("/predict", json=customer_data)
    assert response.status_code == 422

def test_predict(customer_data):
    response = client.post("/predict", json=customer_data)

    assert response.status_code == 200

    result = response.json()
    assert 0 <= result["churn_probability"] <= 1
    assert result["prediction"] in ["Yes", "No"]
    assert result["risk_level"] in ["Low", "Medium", "High"]

    if result["churn_probability"] >= threshold:
        assert result["prediction"] == "Yes"
    else:
        assert result["prediction"] == "No"

    if result["churn_probability"] >= 0.6:
        assert result["risk_level"] == "High"
    elif result["churn_probability"] < 0.3:
        assert result["risk_level"] == "Low"
    else:
        assert result["risk_level"] == "Medium"

@pytest.mark.parametrize(
    "invalid_charge",
    [-1, -100, -9999]
)
def test_invalid_monthly_charges(customer_data, invalid_charge):
    
    customer_data["monthly_charges"] = invalid_charge
    response = client.post("/predict", json=customer_data)

    assert response.status_code == 422

