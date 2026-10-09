# Customer Churn Prediction

Machine Learning project for predicting customer churn using a Random Forest classifier.

The project covers the complete ML workflow: data analysis, preprocessing, model training, evaluation, inference API development, automated testing, and Docker deployment.

## 1. Project Overview

The goal is to predict whether a telecommunications customer is likely to churn based on customer characteristics, subscribed services, contract type, and monthly charges.

The trained model is accessible through a REST API built with FastAPI.

## 2. Dataset

- **Dataset type:** Synthetic educational dataset
- **Number of customers:** 120
- **Number of columns:** 14
- **Target variable:** `churn` (`Yes` / `No`)

The dataset contains customer information, service subscriptions, contract details, and monthly charges.

**Important:** The dataset is synthetic and should not be confused with the original IBM Telco Customer Churn dataset.

## 3. Exploratory Data Analysis

The analysis includes:

- Customer churn distribution
- Churn rates across contract types
- Monthly charges comparison
- Categorical feature analysis
- Data preprocessing and feature encoding

The exploratory analysis is available in `notebooks/02_data_exploration.ipynb`.

## 4. Model Training

The following algorithms were explored:

- Logistic Regression
- Decision Tree
- Random Forest

The final model is a Random Forest classifier.

Preprocessing and model inference are combined using scikit-learn Pipeline and ColumnTransformer.

Hyperparameters were selected using GridSearchCV with cross-validation.

Selected model configuration:
- n_estimators: 200
- max_depth: 5
- Classification threshold: 0.3318

A classification threshold was also selected using cross-validated training predictions.

## 5. Model Evaluation

Performance on the held-out test set (24 customers):

| Metric | Score |
|---|---|
| Accuracy | 0.917 |
| Precision (Churn) | 0.857 |
| Recall (Churn) | 0.857 |
| F1-score (Churn) | 0.857 |

Confusion matrix:

| | Predicted No | Predicted Yes |
|---|---:|---:|
| Actual No | 16 | 1 |
| Actual Yes | 1 | 6 |

These results are preliminary because the dataset is small and synthetic. They should not be interpreted as evidence of real-world predictive performance.

## 6. Technology Stack

- Python 3.14
- pandas and NumPy
- scikit-learn
- Joblib
- FastAPI and Pydantic
- pytest
- Docker
- Git

## 7. Project Structure

```text
customer-churn-ml/
├── data/
│   └── raw/
├── models/
│   └── churn_model.joblib
├── notebooks/
│   ├── 01_environment_test.ipynb
│   └── 02_data_exploration.ipynb
├── src/
│   ├── api.py
│   └── predict.py
├── tests/
│   └── test_api.py
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

## 8. Run with Docker

Build the Docker image:

```bash
docker build -t customer-churn-api .
```

Run the container:

```bash
docker run --rm -p 8000:8000 --name churn-api customer-churn-api
```

Open the interactive API documentation:

http://localhost:8000/docs

### Run locally

Run the following commands from the project root directory (`customer-churn-ml/`).

**1. Create a virtual environment:**

```bash
python -m venv .venv
```

**2. Activate the virtual environment (Windows PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

**3. Install dependencies:**

```bash
python -m pip install -r requirements.txt
```

**4. Start the FastAPI server:**

```bash
python -m uvicorn src.api:app --reload
```

**5. Open the API documentation:**

http://localhost:8000/docs

The API will be available at `http://localhost:8000`.

**6. Run automated tests (optional):**

Install testing dependencies:

```bash
python -m pip install pytest httpx
```

Run the tests:

```bash
python -m pytest -v
```

## 9. API Usage

### POST /predict

Example request:

```json
{
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
```

Example response:

```json
{
  "churn_probability": 0.5137006496088921,
  "prediction": "Yes",
  "risk_level": "Medium"
}
```

The API returns the predicted probability of churn, the binary prediction, and a risk category.

Risk categories use demonstration thresholds and are not independently calibrated.

## 10. Testing

Run the automated tests from the project root:

```bash
python -m pytest -v
```

The project includes 9 automated API tests covering
successful predictions and invalid input validation.

For local testing, install the dependencies from `requirements.txt` and install `pytest` and `httpx` if they are not already available.

## 11. Limitations and Future Improvements

- Replace the synthetic dataset with a larger real-world dataset.
- Evaluate performance on an untouched external test set.
- Evaluate probability calibration.
- Add more comprehensive API tests.
- Improve dependency reproducibility.
- Add automated CI checks using GitHub Actions.

## 12. Project Status

The project currently includes a trained ML pipeline, a working FastAPI service, automated API tests, and a Docker image with a health check.