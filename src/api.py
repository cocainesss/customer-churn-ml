from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
from typing import Literal
from src.predict import predict_churn

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn",
    version="2.0.0"
)


class Customer(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


@app.post("/predict")
def predict(customer: Customer):
    customer_dict = customer.model_dump()
    customer_df = pd.DataFrame([customer_dict])
    result = predict_churn(customer_df)
    return result


@app.get("/")
def home():
    return {
        "message": "Customer Churn API is running"
    }