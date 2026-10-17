# Customer Churn Prediction

A machine learning project that predicts whether a bank customer is likely to leave (churn), along with an interactive Streamlit dashboard for real-time risk scoring and retention recommendations.

## Problem Statement

The bank wants to minimize customer attrition by identifying customers likely to discontinue their relationship with it. Using historical customer data, this project builds a predictive model that enables proactive engagement and personalized retention efforts.

## Project Structure

```
.
├── Churn_Modelling.csv                              # Dataset (10,000 customers)
├── Python_Implementation_for_churn_prediction.ipynb # Data prep, training, evaluation
├── app.py                                           # Streamlit dashboard
├── random_forest_churn_model.pkl                    # Trained Random Forest model
├── scaler.pkl                                       # Fitted StandardScaler
└── README.md
```

## Dataset

`Churn_Modelling.csv` contains 10,000 bank customers with no missing values.

| Column | Description |
|---|---|
| `CreditScore` | Customer credit score |
| `Geography` | Country (France, Germany, Spain) |
| `Gender` | Male / Female |
| `Age` | Customer age |
| `Tenure` | Years as a customer |
| `Balance` | Account balance |
| `NumOfProducts` | Number of bank products held |
| `HasCrCard` | Has a credit card (1/0) |
| `IsActiveMember` | Active member (1/0) |
| `EstimatedSalary` | Estimated annual salary |
| `Exited` | **Target**: 1 = churned, 0 = retained |

`RowNumber`, `CustomerId`, and `Surname` are identifiers and are not used as features.

## Methodology

1. **Load & inspect**: check for missing values (none found).
2. **Encoding**: one-hot encode `Geography` and `Gender` (`drop_first=True`), producing `Germany`, `Spain`, and `Male`.
3. **Features**: 11 inputs: `CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Germany, Spain, Male`.
4. **Split**: 80% train / 20% test (`random_state=20`).
5. **Scaling**: `StandardScaler` fit on the training set only.
6. **Model**: `RandomForestClassifier(n_estimators=150, criterion='gini', random_state=42)`.
7. **Persist**: model and scaler saved as pickle files for reuse in the app.

## Results

Evaluated on the 2,000-row test set:

| Metric | Value |
|---|---|
| Accuracy | **87.25%** |
| Precision (churn) | 0.76 |
| Recall (churn) | 0.47 |
| F1-score (churn) | 0.58 |

**Confusion matrix**

|  | Predicted Retained | Predicted Churned |
|---|---|---|
| **Actual Retained** | 1568 | 55 |
| **Actual Churned** | 200 | 177 |

**Top feature importances:** Age (0.243), EstimatedSalary (0.144), Balance (0.143), CreditScore (0.141), NumOfProducts (0.127).

> **Note:** The model catches under half of actual churners (recall 0.47) because the classes are imbalanced (~19% churn). See [Future Improvements](#future-improvements).

## Streamlit Dashboard

`app.py` provides the "Churn Risk Intelligence Hub":

- Input form for demographics and account details (gender, country, age, salary, credit score, balance, tenure, products, card, activity)
- Preset test profiles (high-risk Germany / low-risk France) in the sidebar
- Predicted exit probability, a risk tier, and a progress indicator
- Suggested retention actions (high-touch outreach vs. cross-sell/monitoring)

## Getting Started

### Prerequisites

- Python 3.9+

### Installation

```bash
pip install streamlit scikit-learn pandas numpy matplotlib jupyter
```

> For reliable pickle loading, use the same scikit-learn version that was used for training.

### Run the dashboard

```bash
streamlit run app.py
```

Make sure `random_forest_churn_model.pkl` and `scaler.pkl` are in the same directory as `app.py`.

### Retrain the model

Open and run all cells in the notebook:

```bash
jupyter notebook Python_Implementation_for_churn_prediction.ipynb
```

This regenerates the `.pkl` files.

## Future Improvements

- Address class imbalance (SMOTE, `class_weight='balanced'`, or threshold tuning) to improve churn recall
- Hyperparameter tuning with cross-validation
- Compare against Gradient Boosting / XGBoost / Logistic Regression
- Add ROC-AUC and precision-recall curves
- Replace the generic "Primary Drivers" text in the app with per-prediction explanations (e.g., SHAP)
- Pin dependency versions in a `requirements.txt`

## Tech Stack

Python · pandas · NumPy · scikit-learn · Matplotlib · Streamlit
