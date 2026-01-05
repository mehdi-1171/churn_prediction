# Telco Customer Churn

## Project Objective

The main objective of this project is to predict customer churn in a telecommunications company using machine learning models.  
By identifying customers who are at high risk of churn, businesses can take proactive actions such as targeted retention campaigns, pricing adjustments, or service improvements.



## Dataset Structure

Rows: Customers

Columns: Customer attributes, services, billing, and churn status

Target Variable: Churn

### Column Descriptions

#### Customer Information:

- customerID : Unique identifier for each customer. This column is used only for identification and is not predictive.

- gender: Gender of the customer (Male, Female).

- SeniorCitizen: Indicates whether the customer is a senior citizen: (1 = Yes, 0 = No)

- Partner : Whether the customer has a partner (Yes, No).

- Dependents : Whether the customer has dependents (Yes, No).

#### Account Information

- tenure: Number of months the customer has been with the company. Customers with lower tenure are more likely to churn.

- Contract: Type of contract the customer has:
    - Month-to-month
    - One year
    - Two year

- PaperlessBilling:  Whether the customer uses paperless billing (Yes, No).

- PaymentMethod:Customer’s payment method:
    - Electronic check
    - Mailed check
    - Bank transfer (automatic)
    - Credit card (automatic)

#### Services Subscribed

- PhoneService: Indicates whether the customer has phone service (Yes, No).

- MultipleLines: Whether the customer has multiple phone lines (Yes, No, No phone service).

- InternetService: Type of internet service:
    - DSL
    - Fiber optic
    - No

- OnlineSecurity: Whether the customer has online security service (Yes, No, No internet service).

- OnlineBackup: Whether the customer has online backup service (Yes, No, No internet service).

- DeviceProtection : Whether the customer has device protection service (Yes, No, No internet service).

- TechSupport: Whether the customer has technical support service (Yes, No, No internet service).

- StreamingTV: Whether the customer has streaming TV service (Yes, No, No internet service).

- StreamingMovies: Whether the customer has streaming movies service (Yes, No, No internet service).

#### Billing Information

- MonthlyCharges: The amount charged to the customer on a monthly basis.
- TotalCharges: The total amount charged to the customer since the start of their subscription. This feature may contain missing or blank values and typically requires preprocessing before analysis.


#### Target Variable
- Churn: Indicates whether the customer has left the company:
  - Yes = Customer churned
  - No = Customer retained

#### Feature Types

- Numerical Features: tenure, MonthlyCharges, TotalCharges, SeniorCitizen

- Categorical Features:
gender, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod

## Project Structure

- 🧹 Prepare Data
- 🧩 Feature Engineering
- 🔧 Logistic Regression Model
- 🔧 Random Forest Model
- 🔧 XGBoost Classification Model


## Use Cases
- Exploratory Data Analysis (EDA)
- Feature importance analysis
- Customer churn prediction
- Business insights

## 🚀 Model

### 📌 1. LogisticRegression
We Develop a **Logistic Regression** model for this data and give this result:
Logistic Regression Results (Default Threshold = 0.3)
- Classification Report: 
 
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| 0     | 0.89      | 0.75   | 0.82     | 1035   |
| 1     | 0.52      | 0.75   | 0.61     | 374    |

**Accuracy:** 0.75  
**Macro Avg:** 0.71  
**Weighted Avg:** 0.79  
**ROC-AUC:** 0.8408

Logistic Regression Results (Default Threshold = 0.5)

| Class | Precision | Recall | F1-Score | Support |
|------|----------|--------|----------|--------|
| 0 (No Churn) | 0.85 | 0.89 | 0.87 | 1035 |
| 1 (Churn) | 0.64 | 0.55 | 0.59 | 374 |

**Accuracy:** 0.80  
**Macro Avg (P / R / F1):** 0.74 / 0.72 / 0.73  
**Weighted Avg (P / R / F1):** 0.79 / 0.80 / 0.79  
**ROC-AUC:** 0.8408

---------------------------------------------------


###  📌 2. Random Forest
#### RF_1:
I work on RF model and get this result:
-  Random Forest Results (Threshold = 0.5)
-  Classification Report

| Class | Precision | Recall | F1-Score | Support |
|------|----------|--------|----------|--------|
| 0 (No Churn) | 0.83 | 0.89 | 0.86 | 1035 |
| 1 (Churn) | 0.61 | 0.49 | 0.55 | 374 |

**Accuracy:** 0.78  
**Macro Avg (P / R / F1):** 0.72 / 0.69 / 0.70  
**Weighted Avg (P / R / F1):** 0.77 / 0.78 / 0.77

**ROC-AUC:** 0.83

- Note:
> Random Forest was evaluated as a non-linear model; however, it did not outperform Logistic Regression in terms of ROC-AUC and Churn Recall, likely due to the sparse and mostly linear nature of the dataset. Therefore, it was excluded from further consideration.

#### RF_2:
We use Grid Search cross-validation for find best Parameter for Data:
- BEST PARAMETER:
> class_weight={0: 1, 1: 2}
>
> min_samples_leaf=20
> 
> min_samples_split=10
> 
> n_estimators=500
> 
> random_state=42


- Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|----------|--------|----------|--------|
| 0 (No Churn) | 0.89 | 0.80 | 0.84 | 1035 |
| 1 (Churn)    | 0.56 | 0.72 | 0.63 | 374 |

**Accuracy:** 0.78  
**Macro Avg (P / R / F1):** 0.73 / 0.76 / 0.74  
**Weighted Avg (P / R / F1):** 0.80 / 0.78 / 0.79  

**ROC-AUC:** 0.85

---------------------------------------------------

### 📌 3. XGBoost Regression

- Classification Report with Threshold (0.4)

| Class | Precision | Recall | F1-Score | Support |
|-------|----------|--------|----------|--------|
| 0 (No Churn) | 0.94 | 0.62 | 0.74 | 1035 |
| 1 (Churn)    | 0.46 | 0.88 | 0.60 | 374 |

**Accuracy:** 0.69  
**Macro Avg (P / R / F1):** 0.70 / 0.75 / 0.67  
**Weighted Avg (P / R / F1):** 0.81 / 0.69 / 0.71  

**ROC-AUC:** 0.85

## Model Comparison Summary

| Model | Threshold | Recall (Churn) | Precision (Churn) | ROC-AUC | Accuracy |
|------|----------|----------------|-------------------|--------|----------|
| Logistic Regression | 0.30 | 0.75 | 0.52 | 0.841 | 0.75 |
| Random Forest (Tuned) | 0.50 | 0.72 | 0.56 | 0.847 | 0.78 |
| XGBoost (Tuned) | 0.40 | **0.88** | 0.46 | 0.847 | 0.69 |

- Logistic Regression provides a strong baseline with stable performance and high interpretability.
- Random Forest captures non-linear patterns but does not significantly outperform Logistic Regression.
- XGBoost achieves the highest Churn recall, making it the best choice when identifying at-risk customers is the top priority.
- Lowering the decision threshold improves Churn recall at the cost of overall accuracy, which is acceptable in churn prediction use cases.

## Evaluation Strategy

Due to class imbalance in the dataset (fewer churned customers), model evaluation focuses not only on accuracy but also on:

- Recall for the Churn class (Class = 1)
- ROC-AUC score to measure overall discrimination ability
- Precision–Recall trade-offs based on different decision thresholds

In churn prediction problems, higher recall is often prioritized to minimize the number of missed churn customers.


## 🔍 Feature Engineering

To improve model performance and better capture customer behavior, several feature transformations and encodings were applied:

- Binary encoding for Yes/No categorical features
- One-Hot Encoding for multi-class categorical variables such as:
  - Contract
  - InternetService
  - PaymentMethod
- Numerical scaling using Min-Max normalization for models sensitive to feature scale (e.g., Logistic Regression)

These steps help reduce model bias and improve learning efficiency, especially for linear models.

## 🔥 Final Conclusion

- Logistic Regression provides a strong and interpretable baseline with stable performance.
- Random Forest, despite modeling non-linear relationships, does not significantly outperform Logistic Regression on this dataset.
- XGBoost achieves the highest recall for churned customers, making it the most suitable model when the business goal is churn prevention.
- Adjusting the classification threshold plays a critical role in balancing recall and precision for churn prediction tasks.

Overall, XGBoost with a lower decision threshold is recommended for identifying high-risk customers in real-world churn management scenarios.


## Future Work

- Advanced feature engineering based on customer behavior patterns
- SHAP analysis for better model interpretability
- Cost-sensitive learning based on business impact
- Deployment of the model as a REST API

## License

This dataset is publicly available and commonly used for educational and research purposes.