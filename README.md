# Financial_Fraud_Detection
Classify financial transactions as fraudulent or non-fraudulent
# Fraud Detection in Financial Transactions

**Author:** Dipti Srivastava

### Executive Summary

This project uses supervised machine learning to predict whether a financial transaction is fraudulent. The goal is to build models that can assist financial institutions in identifying suspicious activity with high precision and recall — ultimately reducing fraud losses while minimizing false positives for legitimate users.

### Rationale

Financial fraud is a growing global concern, costing billions each year. By leveraging machine learning, businesses can proactively identify patterns in fraudulent behavior and intervene earlier. The challenge lies in detecting rare fraudulent cases hidden among massive volumes of legitimate transactions, with minimal disruption to customer experience.


### Research Question

> Can we accurately classify financial transactions as fraudulent or non-fraudulent using machine learning models?

### Data Sources

- **IEEE-CIS Fraud Detection dataset**  
  [Kaggle Link](https://www.kaggle.com/competitions/ieee-fraud-detection)
  
This dataset includes anonymized transaction data with engineered features including device type, card information, and email domains. The target variable is `isFraud`.

### Methodology

#### 1. Exploratory Data Analysis (EDA)

- Assessed dataset shape, data types, and cardinality of features.
- Visualized missing data using heatmaps and computed missing value % per feature.
- Identified target imbalance: fraud (`isFraud=1`) accounts for ~0.3% of all transactions.
- Used histograms, bar plots, and fraud rate heatmaps to explore patterns in:
  - `ProductCD`, `DeviceType`, `card4`, `card6`
  - Email domains and operating systems
  - Temporal patterns using `TransactionDT` for transaction hour/day analysis

#### 2. Feature Engineering

- Created time-based features:
  - `TransactionHour`, `TransactionDay`, and `TransactionWeekday` from `TransactionDT`
- Added missingness flags for key identity fields:
  - `id_30_missing_flag`, `id_31_missing_flag`, etc.
- Applied `LabelEncoder` on high-signal categorical variables:
  - `ProductCD`, `card4`, `card6`, `DeviceType`, `id_30`, `id_31`, `P_emaildomain`, `R_emaildomain`
- Pruned low-signal and high-missing columns using thresholds:
  - >90% missing AND low variance AND no fraud signal

#### 3. Modeling

- **Class imbalance handling**:
  - Used **SMOTE** oversampling for Logistic Regression
  - Used **class_weight='balanced'** for Random Forest
- **Models evaluated**:
  - Logistic Regression with `StandardScaler` (for convergence)
  - Random Forest without scaling (trees are scale-invariant)
- **Train/test split**:
  - 70/30 stratified split to preserve fraud ratio in both sets
- **Feature preprocessing**:
  - Imputed missing values with `SimpleImputer(strategy='median')`
  - Standardized numerical features for Logistic Regression using `StandardScaler`

#### 4. Evaluation

- Computed:
  - Accuracy, Precision, Recall, F1-score, ROC-AUC
  - Weighted and macro averages to reflect class imbalance
- Visualized:
  - Confusion matrices for both models
  - Summary tables comparing fraud detection performance
- Business-focused trade-off analysis:
  - Logistic Regression: better **recall**, catches more fraud
  - Random Forest: better **precision**, fewer false alarms

> These evaluations informed model trade-offs depending on business use case:
> - High recall (fraud surveillance) vs. high precision (automated blocking)

### Results

| Model               | Precision (Fraud) | Recall (Fraud) | ROC-AUC |
|--------------------|-------------------|----------------|---------|
| Logistic Regression (SMOTE) | 8%               | **61%**         | 0.74    |
| Random Forest (weighted)    | **92%**           | 31%            | **0.89** |

- Logistic Regression catches more fraud but with high false positives.
- Random Forest is highly precise but less sensitive.
- Model choice depends on business risk tolerance and customer experience trade-offs.


### Next Steps

- Add advanced models: **XGBoost**, **LightGBM**, **CatBoost**
- Perform **hyperparameter tuning** and **cross-validation**
- Apply **feature importance** and **SHAP explainability**
- Build a **threshold optimization dashboard** using precision-recall curves
- Package into a deployable pipeline (e.g., Flask API, cloud integration)


### Outline of Project

- [Notebook 1 – EDA & Feature Engineering](#)
- [Notebook 2 – Baseline Modeling](#)
- [Notebook 3 – Evaluation & Interpretation](#)

> Replace `#` with your actual GitHub notebook URLs.


### Contact and Further Information

For more information or collaboration, please contact:  
📧 [your-email@example.com]  
🌐 [LinkedIn](https://linkedin.com/in/yourprofile)  
🐦 [Twitter](https://twitter.com/yourhandle)


*Built as part of a capstone project to showcase applied ML skills in a high-impact financial context.*
