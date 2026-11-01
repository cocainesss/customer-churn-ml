# Customer Churn Prediction

An end-to-end Machine Learning project for predicting telecommunications customer churn using a Random Forest classifier.

The project covers exploratory data analysis, preprocessing, model training, hyperparameter tuning, model evaluation, REST API development, automated testing, Docker containerization, and GitHub Actions CI.

## 1. Project Overview

The goal is to predict whether a telecommunications customer is likely to churn based on demographic information, subscribed services, contract details, tenure, and billing information.

The final model is deployed through a REST API built with FastAPI. The API returns a churn probability, a binary prediction, and a risk category.

## 2. Dataset

The project uses the publicly available **IBM Telco Customer Churn sample dataset**.

- **Number of customers:** 7,043
- **Number of original columns:** 21
- **Number of model input features:** 19
- **Target variable:** `Churn` (`Yes` / `No`)
- **Churn rate:** approximately 26.54%

Dataset source: [IBM Telco Customer Churn CSV](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)

The dataset includes customer demographics, service subscriptions, contract types, payment methods, monthly charges, and total charges.

**Note:** This is a public educational sample dataset, not private production customer data.

### Data preprocessing

- Removed `customerID` from model inputs.
- Converted `TotalCharges` to numeric values.
- Filled 11 missing `TotalCharges` values with zero for customers with zero tenure.
- Split the dataset into training (80%) and test (20%) sets using stratification.
- Applied `StandardScaler` to numerical features.
- Applied `OneHotEncoder(handle_unknown="ignore")` to categorical features.

Preprocessing is integrated into a scikit-learn `Pipeline` using `ColumnTransformer`.

## 3. Exploratory Data Analysis

The exploratory analysis covers:

- Customer churn distribution
- Churn patterns across contract types
- Customer tenure
- Monthly and total charges
- Service subscriptions
- Categorical feature relationships

The updated analysis and model experiments are available in:

`notebooks/03_real_data_exploration.ipynb`

Earlier exploratory notebooks are retained as part of the project's development history.

## 4. Model Training

The updated dataset was used to evaluate:

- Logistic Regression
- Random Forest

The final deployed model is a **Random Forest classifier**.

Hyperparameters were tuned using `GridSearchCV` with five-fold cross-validation and churn-class F1-score as the optimization metric.

### Selected Random Forest configuration

| Parameter | Value |
|---|---|
| `n_estimators` | 50 |
| `max_depth` | 10 |
| `min_samples_leaf` | 1 |
| `random_state` | 42 |
| Classification threshold | 0.30 |

The selected hyperparameter configuration achieved a cross-validation F1-score of approximately **0.5781** during grid search.

The classification threshold was selected using cross-validated predictions on the training set to prioritize churn detection and improve the balance between precision and recall.

## 5. Model Evaluation

The test set contains **1,409 customers**.

### Model comparison

| Model | Threshold | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.50 | 0.8055 | 0.6572 | 0.5588 | 0.6040 |
| Logistic Regression | 0.30 | 0.7495 | 0.5193 | 0.7540 | 0.6150 |
| Tuned Random Forest | 0.50 | 0.7970 | 0.6457 | 0.5214 | 0.5769 |
| Tuned Random Forest | 0.30 | 0.7509 | 0.5206 | 0.7754 | 0.6230 |

Precision, recall, and F1-score refer to the positive churn class (`Yes`).

### Final deployed model

The deployed Random Forest uses a classification threshold of 0.30.

| Metric | Score |
|---|---:|
| Accuracy | 0.7509 |
| Precision (Churn) | 0.5206 |
| Recall (Churn) | 0.7754 |
| F1-score (Churn) | 0.6230 |

Confusion matrix:

| | Predicted No | Predicted Yes |
|---|---:|---:|
| Actual No | 768 | 267 |
| Actual Yes | 84 | 290 |

The lower classification threshold increases recall, allowing the model to identify more customers who churn, at the cost of additional false positives.

The final Random Forest slightly outperformed the tested Logistic Regression configuration in churn-class F1-score. This difference is small and should not be interpreted as proof of general superiority.

### Feature importance

Permutation importance analysis identified several influential features, including:

- Contract
- InternetService
- tenure
- OnlineSecurity
- MonthlyCharges
- TotalCharges

Permutation importance was calculated on the test set using the default 0.50 prediction threshold, rather than the deployed 0.30 threshold.

**Evaluation limitation:** The test set was inspected during multiple model experiments. Consequently, the reported test metrics should be treated as exploratory rather than as a fully independent final generalization estimate.

## 6. Technology Stack

- Python 3.14
- pandas and NumPy
- scikit-learn
- Joblib
- FastAPI and Pydantic
- pytest and HTTPX
- Docker
- Git and GitHub
- GitHub Actions

## 7. Project Structure

```text
customer-churn-ml/
├── .github/
│   └── workflows/
│       └── tests.yml
├── data/
│   └── raw/
│       └── telco_churn_real.csv
├── models/
│   ├── churn_model.joblib
│   ├── churn_model_real.joblib
│   └── churn_threshold_real.joblib
├── notebooks/
│   ├── 01_environment_test.ipynb
│   ├── 02_data_exploration.ipynb
│   └── 03_real_data_exploration.ipynb
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

The earlier model artifact is retained for historical reference. The current API uses `churn_model_real.joblib` and `churn_threshold_real.joblib`.

## 8. Installation and Deployment

### Run with Docker

Build the Docker image:

```bash
docker build -t customer-churn-api:v2 .
```

Run the container:

```bash
docker run --rm -p 8000:8000 customer-churn-api:v2
```

Open the interactive API documentation:

http://localhost:8000/docs

The Docker image includes the trained model and a health check for the API root endpoint.

### Run locally

Run these commands from the project root directory.

**1. Create a virtual environment:**

```bash
python -m venv .venv
```

**2. Activate it (Windows PowerShell):**

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

## 9. API Usage

### GET /

Returns the API status.

Example response:

```json
{
  "message": "Customer Churn API is running"
}
```

### POST /predict

Accepts customer information and returns a churn prediction.

Example request:

```json
{
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
```

Example response from the deployed model:

```json
{
  "churn_probability": 0.7855181541804545,
  "prediction": "Yes",
  "risk_level": "High"
}
```

### Prediction logic

The API predicts churn when:

`churn_probability >= 0.30`

Risk categories:

| Probability | Risk level |
|---|---|
| Below 0.30 | Low |
| 0.30 to below 0.60 | Medium |
| 0.60 and above | High |

These risk categories are demonstration rules and have not been independently calibrated against business outcomes.

## 10. Automated Testing

The project includes **12 automated API test cases** covering:

- API availability
- Successful churn predictions
- Prediction threshold consistency
- Risk category consistency
- Invalid contract values
- Negative monthly charges
- Negative total charges
- Missing required input fields

Run the tests:

```bash
python -m pip install pytest httpx
python -m pytest -v
```

Latest verified local result: **12 passed**.

## 11. Continuous Integration

GitHub Actions automatically runs the workflow on pushes and pull requests targeting the `main` branch.

The workflow contains two jobs:

1. **test** — installs dependencies and runs automated API tests.
2. **docker-build** — builds the Docker image after the test job succeeds.

The updated workflow has successfully completed both jobs on GitHub Actions.

## 12. Limitations and Future Improvements

Current limitations:

- The dataset is an educational sample rather than production data.
- The test set was used during exploratory model comparisons.
- Predicted probabilities have not been independently calibrated.
- Risk categories use fixed demonstration thresholds.
- Some Python dependencies are not fully version-pinned.
- The Docker image is built in CI but not yet subjected to an automated container-level inference test.

Planned improvements:

- Use a fresh, untouched holdout set or nested cross-validation for more rigorous model selection and evaluation.
- Evaluate probability calibration.
- Introduce business-oriented metrics and cost-sensitive threshold selection.
- Add stricter input consistency validation.
- Add Docker integration tests.
- Improve dependency reproducibility.
- Add model monitoring and data drift checks.

## Project Status

**Completed:** ML pipeline, hyperparameter tuning, threshold optimization, model serialization, FastAPI inference, automated tests, Docker deployment, and GitHub Actions CI.

The project is suitable as an educational end-to-end ML engineering portfolio project. It is not presented as a production-validated churn prediction system.
