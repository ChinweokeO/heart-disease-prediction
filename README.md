# Heart Disease Prediction MLOps Pipeline

![Python](https://img.shields.io/badge/python-3.10+-blue)
![MLflow](https://img.shields.io/badge/MLflow-experiment_tracking-blue)
![DVC](https://img.shields.io/badge/DVC-data_versioning-orange)
![Monitoring](https://img.shields.io/badge/Monitoring-Evidently-green)


## Overview

This project implements an end-to-end MLOps pipeline for predicting the presence of heart disease using clinical patient data. The goal is not only to train a machine learning model, but also to demonstrate production-ready machine learning practices including data versioning, experiment tracking, automated testing, continuous integration, and model monitoring.

The project uses a Heart Disease dataset containing demographic, clinical, and diagnostic measurements to classify whether a patient has heart disease. The pipeline is designed to be reproducible, maintainable, and extensible, following industry-standard MLOps workflows.

---

## Problem Statement

Heart disease remains one of the leading causes of mortality worldwide. Early identification of individuals at risk can support timely intervention and improve health outcomes.

This project builds a classification model that predicts the presence of heart disease based on patient characteristics such as age, cholesterol levels, blood pressure, chest pain type, electrocardiogram results, and other clinical indicators.

---

## Dataset

### Source

Heart Disease Dataset (UCI Machine Learning Repository)

### Target Variable

**num**

* 0 = No heart disease
* 1 = Presence of heart disease

### Features

| Feature  | Description                                    |
| -------- | ---------------------------------------------- |
| age      | Age in years                                   |
| sex      | Biological sex                                 |
| cp       | Chest pain type                                |
| trestbps | Resting blood pressure                         |
| chol     | Serum cholesterol                              |
| fbs      | Fasting blood sugar                            |
| restecg  | Resting electrocardiographic results           |
| thalach  | Maximum heart rate achieved                    |
| exang    | Exercise-induced angina                        |
| oldpeak  | ST depression induced by exercise              |
| slope    | Slope of peak exercise ST segment              |
| ca       | Number of major vessels colored by fluoroscopy |
| thal     | Thalassemia category                           |

---

## Project Structure

```text
heart-disease-prediction/
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── heart_disease.csv
│
│
├── reports/
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluation.py
│   ├── compare_experiments.py
│   └── monitor_drift.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_data_validation.py
│   └── test_model_validation.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* MLflow
* DVC
* Pytest
* GitHub Actions
* Evidently
* YAML

---

## MLOps Components

### Data Versioning (DVC)

The dataset is tracked using DVC rather than Git to enable reproducible experimentation and efficient version control of large files.

### Experiment Tracking (MLflow)

Each training run logs:

* Model hyperparameters
* Dataset version
* Evaluation metrics
* Trained model artifacts

Experiments can be compared programmatically using `compare_experiments.py`.

### Automated Testing (Pytest)

The project includes:

#### Preprocessing Tests

* Missing value handling
* Categorical encoding validation
* Data integrity checks
* Error handling validation

#### Data Validation Tests

* Required columns exist
* Target values are valid
* Numeric feature ranges are reasonable

#### Model Validation Tests

* Prediction shape verification
* Minimum performance threshold validation

### Continuous Integration (GitHub Actions)

The CI/CD pipeline automatically:

1. Installs dependencies
2. Executes the test suite
3. Trains the model
4. Verifies model performance requirements

### Drift Monitoring (Evidently)

The monitoring pipeline compares reference training data against simulated production data to detect feature drift and identify when retraining may be necessary.

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd heart-disease-prediction
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

### Windows

```bash
venv\Scripts\activate
```

### macOS/Linux

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running Training

Train the model:

```bash
python src/train.py
```

The training script:

* Loads configuration values from `config.yaml`
* Preprocesses the data
* Trains the model
* Evaluates performance
* Logs metrics and artifacts to MLflow

---

## Running Tests

Run all tests:

```bash
pytest tests/ -v
```

---

## MLflow Experiment Tracking

Launch the MLflow UI:

```bash
mlflow ui
```

Then navigate to:

```text
http://localhost:5000
```

View:

* Experiment runs
* Parameters
* Metrics
* Model artifacts

---

## Comparing Experiments

Identify the best-performing experiment:

```bash
python src/compare_experiments.py
```

The script queries MLflow runs and returns the highest-performing model based on the primary evaluation metric.

---

## Drift Monitoring

Run drift detection:

```bash
python src/monitor_drift.py
```

Outputs:

* Drift summary in terminal
* HTML drift report saved to `reports/`

---

## Future Improvements

* Hyperparameter optimization using Optuna
* Model deployment with FastAPI
* Containerization with Docker
* Cloud deployment using AWS
* Automated retraining workflows
* Real-time monitoring dashboards

---

## Author

**Chinweoke Stephanie Okonkwo**

Data Science | Machine Learning | MLOps | Healthcare Analytics
