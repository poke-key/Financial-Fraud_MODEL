# Financial_Fraud_Detection
Classify financial transactions as fraudulent or non-fraudulent
# Fraud Detection in Financial Transactions

**Author:** Dipti Srivastava

## Executive Summary

This project uses supervised machine learning to predict whether a financial transaction is fraudulent. The goal is to build models that can assist financial institutions in identifying suspicious activity with high precision and recall — ultimately reducing fraud losses while minimizing false positives for legitimate users.

## Rationale

Financial fraud is a growing global concern, costing billions each year. By leveraging machine learning, businesses can proactively identify patterns in fraudulent behavior and intervene earlier. The challenge lies in detecting rare fraudulent cases hidden among massive volumes of legitimate transactions, with minimal disruption to customer experience.


## Research Question

> **Can we accurately classify financial transactions as fraudulent or non-fraudulent using machine learning models?**

### Data Sources

- **IEEE-CIS Fraud Detection dataset**  
  [Kaggle Link](https://www.kaggle.com/competitions/ieee-fraud-detection)
  
This dataset includes anonymized transaction data with engineered features including device type, card information, and email domains. The target variable is `isFraud`.

## Methodology

## I. Exploratory Data Analysis (EDA)

- Assessed dataset shape, data types, and cardinality of features.
- Visualized missing data using heatmaps and computed missing value % per feature.
- Identified target imbalance: fraud (`isFraud=1`) accounts for ~0.3% of all transactions.
- Used histograms, bar plots, and fraud rate heatmaps to explore patterns in:
  - `ProductCD`, `DeviceType`, `card4`, `card6`
  - Email domains and operating systems
  - Temporal patterns using `TransactionDT` for transaction hour/day analysis

## Initial EDA Findings based on RAW data

### Key Observations

1. **Target Variable Imbalance (`isFraud`)**
   - Highly imbalanced: ~99.8% non-fraud vs ~0.2% fraud.
   - Will require class balancing strategies (e.g., SMOTE or `class_weight='balanced'`).

   <img src="images/Fraud-vs-NonFraud.png" alt="Target Class Imbalance" width="500"/>

2. **Missing Values**
   - Over 200 features have missing data, especially among `id_`, `D_`, and `V_` columns.
   - Some features (e.g., `id_12`, `id_13`, `V300+`) have >90% missing values.
   - Missingness matrix reveals structured patterns—some columns may be worth keeping despite missingness.
     
![Top 50 Missing Values by Column](/images/Top%20N%20Missing%20Values%20by%20Column.png)
3. **Numeric Feature Distributions**
   - `TransactionAmt`, `id_02`, and `D15` show heavy right skew and possible outliers.
   - Fraudulent transactions cluster at lower transaction amounts.
     
![Numeric Feature Distributions](/images/Numeric%20Feature%20Distributions%20by%20Fraud%20Label.png)
4. **Categorical Features**
   - Some categorical variables (`ProductCD`, `card4`, `card6`) show variation by fraud class.
   - Fields like `DeviceType`, `id_30`, and `id_31` contain `"unknown"` and `"NotFound"` entries—these may be fraud indicators.
   - Some categories are heavily imbalanced, and will require careful encoding.
     
![Categorical Features](/images/Categorical%20Feature%20Distribution%20by%20Fraud%20Label.png)


### Recommendations Based on EDA

| Area                     | Recommendation                                                                 |
|--------------------------|----------------------------------------------------------------------------------|
| Class Imbalance          | Use resampling (e.g., SMOTE) or adjust model class weights.                     |
| High-Missing Columns     | Drop features with >90% missing **and** low variance.                           |
| Categorical Variables    | Use Label Encoding; create binary flags for `"unknown"`/`"NotFound"` entries.   |
| Skewed Numeric Features  | Apply log transformation (e.g., on `TransactionAmt`, `id_02`) where appropriate. |
| Temporal Features        | Engineer features from `TransactionDT`: hour, day, weekday.                     |
| Device/Browser Info      | Investigate `id_30`, `id_31`, and `DeviceType` for fraud patterns.              |

## II. Feature Engineering

- Created time-based features:
  - `TransactionHour`, `TransactionDay`, and `TransactionWeekday` from `TransactionDT`
- Added missingness flags for key identity fields:
  - `id_30_missing_flag`, `id_31_missing_flag`, etc.
- Applied `LabelEncoder` on high-signal categorical variables:
  - `ProductCD`, `card4`, `card6`, `DeviceType`, `id_30`, `id_31`, `P_emaildomain`, `R_emaildomain`
- Pruned low-signal and high-missing columns using thresholds:
  - >90% missing AND low variance AND no fraud signal
1. **Dropped Low-Value Columns**
   - Removed columns with >90% missing **and** only 1 unique value (low variance).
   - This reduced dataset dimensionality and helped eliminate noise.

2. **Handled “Unknown” and Placeholder Values**
   - Replaced string markers like `"unknown"`, `"NotFound"`, and `"nan"` with true `NaN` values.
   - Ensured consistent handling of missing values across both string and numeric features.

3. **Imputed Missing Values and Created Missing Flags**
   - Imputed median values for important numerical features (`TransactionAmt`, `D15`).
   - Created binary indicators (e.g., `id_30_missing_flag`) to preserve missingness patterns — potentially important for detecting fraud.

4. **Encoded Categorical Variables**
   - Used Label Encoding for features like `ProductCD`, `card4`, `card6`, `DeviceType`, `id_30`, and `id_31`.
   - Converted string categories into model-compatible numerical values.

5. **Engineered New Features from `TransactionDT`**
   - Converted `TransactionDT` to `TransactionDate`.
   - Extracted temporal features: `TransactionHour`, `TransactionDay`, `TransactionWeekday`.
   - These new features allow the model to detect **fraud patterns based on time-of-day or day-of-week**.

#### Feature Correlation Check
![Feature Correlation](/images/Feature%20Correlation%20Check.png)

**We computed a correlation matrix on all selected features to detect highly correlated (redundant) pairs. A threshold of 0.9 was used to identify problematic correlations.**

**Result: No pairs exceeded 0.9 correlation.**

This means:
- All selected features provide unique or complementary signal.
- We will retain the following feature set for baseline modeling.

### Final Feature Set for Modeling

| Feature Name            | Feature Type       | Notes/Description                                  |
|--------------------------|--------------------|----------------------------------------------------|
| TransactionAmt           | Numeric             | Transaction amount (skewed; log-transform optional) |
| id_02                    | Numeric             | Distance-related feature; correlated with fraud    |
| D15                      | Numeric             | Temporal distance feature; correlated with fraud   |
| TransactionHour          | Numeric (Engineered) | Hour of transaction extracted from timestamp       |
| TransactionDay           | Numeric (Engineered) | Day of transaction extracted from timestamp        |
| TransactionWeekday       | Numeric (Engineered) | Weekday of transaction extracted from timestamp    |
| ProductCD                | Categorical         | Product code related to transaction type           |
| card4                    | Categorical         | Card type (e.g., Visa, Mastercard)                  |
| card6                    | Categorical         | Card usage type (e.g., debit, credit)               |
| DeviceType               | Categorical         | Device type (e.g., desktop, mobile)                 |
| id_30                    | Categorical         | Device operating system (e.g., Windows, iOS)        |
| id_31                    | Categorical         | Browser type                                       |
| P_emaildomain            | Categorical         | Purchaser email domain                             |
| R_emaildomain            | Categorical         | Recipient email domain                             |
| id_30_missing_flag       | Binary Flag         | Missing flag for `id_30`                            |
| id_31_missing_flag       | Binary Flag         | Missing flag for `id_31`                            |
| id_33_missing_flag       | Binary Flag         | Missing flag for `id_33` (if available)             |


## III. Modeling

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
    
<p align="center">
  <img src="images/Confusion%20Matrix%20-%20LR.png" alt="Confusion Matrix - Logistic Regression" width="500"/>
  <img src="images/Confusion%20Matrix%20-%20RF.png" alt="Confusion Matrix - Random Forest" width="500"/>
</p>

<p align="center"><b>Confusion Matrices: Logistic Regression vs Random Forest</b></p>


## IV. Evaluation

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

**Summary: Baseline Modeling**
We trained and evaluated two baseline models to classify fraudulent transactions:

**Logistic Regression:** Interpretable and effective with SMOTE-balanced training data.
**Random Forest:** Handled imbalanced data with class_weight='balanced'.

**Key Takeaways:**
Logistic Regression performed well with SMOTE it has better recall on minority class.
Random Forest had slightly better ROC-AUC and overall F1 score.
Next steps: Tune hyperparameters, try gradient boosting (XGBoost/LightGBM), and add feature importance analysis

## Results

| Model               | Precision (Fraud) | Recall (Fraud) | ROC-AUC |
|--------------------|-------------------|----------------|---------|
| Logistic Regression (SMOTE) | 8%               | **61%**         | 0.74    |
| Random Forest (weighted)    | **92%**           | 31%            | **0.89** |

## Insights:
- **Logistic Regression** catches more fraud but with many false alarms.
- **Random Forest** is more precise but misses more fraud.
- Logistic is better when **recall matters most** (e.g., catching fraud at all costs).
- Random Forest is better when **precision matters** (e.g., avoid flagging good customers).

## Business Takeaway:
The right model depends on risk tolerance:
- Want to **catch more fraud** even if noisy? Use Logistic Regression.
- Want to **flag fraud only when you're confident?** Use Random Forest.


## Next Steps

- Add advanced models: **XGBoost**
- Perform **hyperparameter tuning** and **cross-validation**
- Apply **feature importance** and **SHAP explainability**
- Build a **threshold optimization dashboard** using precision-recall curves
- Package into a deployable pipeline (e.g., Flask API, cloud integration)


### Outline of Project

- [Notebook – EDA & Feature Engineering](/detecting_financial_fraud.ipynb)

### Contact and Further Information

For more information or collaboration, please contact:

- 📧 [Email -Dipti](mailto:dsrivast@gmail.com)
- 🌐 [LinkedIn – Dipti Srivastava](https://linkedin.com/in/diptishrivastav)
- 🐦 [Twitter – @dsrivast](https://twitter.com/dsrivast)



*Built as part of a capstone project to showcase applied ML skills in a high-impact financial context.*
