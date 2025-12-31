# Telco Customer Churn





## Dataset Structure

Rows: Customers

Columns: Customer attributes, services, billing, and churn status

Target Variable: Churn

#### Column Descriptions

- Customer Information:

- customerID
Unique identifier for each customer. This column is used only for identification and is not predictive.

- gender

Gender of the customer (Male, Female).

- SeniorCitizen

Indicates whether the customer is a senior citizen:
1 = Yes, 0 = No.

Partner
Whether the customer has a partner (Yes, No).

Dependents
Whether the customer has dependents (Yes, No).

Account Information

tenure
Number of months the customer has been with the company. Customers with lower tenure are more likely to churn.

Contract
Type of contract the customer has:

Month-to-month

One year

Two year

PaperlessBilling
Whether the customer uses paperless billing (Yes, No).

PaymentMethod
Customer’s payment method:

Electronic check

Mailed check

Bank transfer (automatic)

Credit card (automatic)

Services Subscribed

PhoneService
Indicates whether the customer has phone service (Yes, No).

MultipleLines
Whether the customer has multiple phone lines (Yes, No, No phone service).

InternetService
Type of internet service:

DSL

Fiber optic

No

OnlineSecurity
Whether the customer has online security service (Yes, No, No internet service).

OnlineBackup
Whether the customer has online backup service (Yes, No, No internet service).

DeviceProtection
Whether the customer has device protection service (Yes, No, No internet service).

TechSupport
Whether the customer has technical support service (Yes, No, No internet service).

StreamingTV
Whether the customer has streaming TV service (Yes, No, No internet service).

StreamingMovies
Whether the customer has streaming movies service (Yes, No, No internet service).

Billing Information

MonthlyCharges
The amount charged to the customer on a monthly basis.

TotalCharges
The total amount charged to the customer since the start of their subscription. This feature may contain missing or blank values and typically requires preprocessing before analysis.

Target Variable

Churn
Indicates whether the customer has left the company:

Yes = Customer churned

No = Customer retained

Feature Types

Numerical Features:
tenure, MonthlyCharges, TotalCharges, SeniorCitizen

Categorical Features:
gender, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod

Use Cases

Exploratory Data Analysis (EDA)

Customer churn prediction

Feature importance analysis

Business insights and retention strategies

Notes

TotalCharges may contain empty strings and should be converted to numeric values during preprocessing.

customerID should be excluded from modeling.

License

This dataset is publicly available and commonly used for educational and research purposes.