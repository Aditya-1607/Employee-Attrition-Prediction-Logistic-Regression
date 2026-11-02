# Employee Attrition Prediction with Logistic Regression

A machine learning project that predicts whether an employee is likely to leave a company using **Logistic Regression** and the IBM HR Analytics Employee Attrition dataset.

---

## Overview

This project builds a binary classification model using selected employee information. The workflow includes loading a local CSV file, selecting features, encoding the target variable, splitting the data, scaling numerical features, training a Logistic Regression model, evaluating its performance, and predicting attrition for a sample employee.

---

## Dataset

- **File:** `employee_attrition.csv`
- **Dataset:** IBM HR Analytics Employee Attrition
- **Target column:** `Attrition`
- **Target values:** `Yes` and `No`, mapped to `1` and `0` respectively.

The dataset is loaded from the CSV file stored in the project folder:

```python
df = pd.read_csv('employee_attrition.csv')
```

---

## Features Used

The model uses four numerical features:

| Feature | Description |
|---|---|
| `Age` | Age of the employee in years |
| `MonthlyIncome` | Monthly income of the employee |
| `TotalWorkingYears` | Total career experience in years |
| `YearsAtCompany` | Number of years the employee has worked at the company |

---

## Workflow

1. **Import libraries** — Import pandas and the required scikit-learn tools.
2. **Load data** — Read `employee_attrition.csv` using pandas.
3. **Select features and target** — Use the four numerical features and the `Attrition` column.
4. **Encode target** — Convert `Yes` to `1` and `No` to `0`.
5. **Split data** — Use an 80/20 training and testing split with `random_state=42`.
6. **Scale and train** — Apply `StandardScaler` followed by `LogisticRegression(max_iter=1000)` using a pipeline.
7. **Evaluate** — Print the accuracy score and confusion matrix.
8. **Predict** — Test the trained model using a custom employee profile.

---

## Model Training

```python
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

model.fit(X_train, y_train)
```

---

## Evaluation

```python
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

The accuracy and confusion matrix are calculated when the notebook is run. Their values are not fixed in this README.

---

## Example Prediction

The notebook tests the model with this sample employee profile:

```python
new_employee = pd.DataFrame({
    'Age': [28],
    'MonthlyIncome': [3500],
    'TotalWorkingYears': [3],
    'YearsAtCompany': [1]
})

prediction = model.predict(new_employee)

result = "Attrition (Will Leave)" if prediction[0] == 1 else "Retained (Will Stay)"

print("Prediction:", result)
```

---

## Requirements

- Python 3.8 or higher
- pandas
- scikit-learn
- Jupyter Notebook (optional)

Install dependencies:

```bash
pip install pandas scikit-learn jupyter
```

---

## Project Structure

```text
├── employee_attrition.csv                 # Dataset
├── employee_attrition_prediction.ipynb    # Main notebook
└── README.md                              # Project documentation
```

---

## Key Takeaways

- Logistic Regression is used for binary classification.
- Employee attrition is encoded as a binary target.
- StandardScaler standardizes the numerical input features.
- Accuracy and the confusion matrix are used to evaluate predictions.
- A custom employee profile can be passed to the trained model.
