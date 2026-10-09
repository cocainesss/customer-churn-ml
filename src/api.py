from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
from typing import Literal
from src.predict import predict_churn

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn",
    version="1.0.0"
)
class Customer(BaseModel):
    gender: Literal["Male", "Female"]
    senior_citizen: Literal[0, 1]
    partner: Literal["Yes", "No"]
    dependents: Literal["Yes", "No"]
    phone_service: Literal["Yes", "No"]
    multiple_lines: Literal["Yes", "No"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    online_security: Literal["Yes", "No"]
    online_backup: Literal["Yes", "No"]
    device_protection: Literal["Yes", "No"]
    contract: Literal["Month-to-month", "One year", "Two year"]
    monthly_charges: float = Field(ge=0, le=1000)

@app.post('/predict')
def predict(customer: Customer):
    customer_dict = customer.model_dump()
    customer_df = pd.DataFrame([customer_dict])
    result = predict_churn(customer_df)
    return result

@app.get('/')
def home():
    return {
        "message": "Customer Churn API is running"
    }
