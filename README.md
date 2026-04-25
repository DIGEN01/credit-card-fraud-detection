# Credit Card Fraud Detection

A three-phase credit card fraud detection project combining Tableau dashboards, Python machine learning, and a fraud risk routing system. Built using real-world transaction data to identify fraudulent patterns and score risk.

## Project Overview

This project was built to solve a common challenge in financial services: how to quickly and accurately spot fraudulent transactions in a stream of millions. The dataset used is the Kaggle Credit Card Fraud dataset, which contains anonymized European cardholder transactions from September 2013. It has 284,807 transactions with only 492 fraudulent cases, making it a heavily imbalanced classification problem.

The project is split into three phases:

1. **Phase 1 - Tableau Dashboard**: Exploratory data analysis and behavioral pattern visualization
2. **Phase 2 - Python Machine Learning**: Feature engineering, model training, and statistical hypothesis testing
3. **Phase 3 - Fraud Risk Routing System**: A scoring and routing logic to triage flagged transactions

## Dataset

| Property | Details |
|----------|----------|
| Source | [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) |
| Records | 284,807 transactions |
| Fraud Cases | 492 (0.17%) |
| Legitimate | 284,315 (99.83%) |
| Features | 28 PCA components + Amount + Time + Class |

> Note: The raw dataset file is not included in this repo due to size. Download it from the Kaggle link above and place it in the `data/` folder before running the notebook.
>
> ## Phase 1 - Tableau Dashboard

The Tableau dashboard (`reports/Fraud_Analysis_Dashboard.twbx`) was built to explore behavioral patterns in the transaction data before any modeling began. The goal was to understand how fraudsters behave differently from regular cardholders across multiple dimensions.

### Key Findings from the Dashboard

- **Transaction timing**: Fraud spikes during late night and early morning hours (around 1-5 AM), suggesting fraudsters prefer low-activity periods when monitoring is minimal
- **Transaction amount**: Fraudulent transactions are typically smaller amounts (median around $9), designed to stay under radar and avoid triggering amount-based alerts
- **Merchant concentration**: A small number of merchants account for a disproportionate share of fraud, indicating potential compromised merchant terminals or coordinated fraud rings
- **Transaction velocity**: Fraudulent cards show bursts of rapid transactions in short time windows, unlike normal spending patterns
- **Geographic patterns**: Certain location clusters show higher fraud density, useful for geo-based risk scoring

- ## Phase 2 - Python Machine Learning

The Jupyter notebook (`credit_card_fraud.ipynb`) contains the full ML pipeline including data preprocessing, feature engineering, model training, evaluation, and statistical testing.

### Feature Engineering

Beyond the 28 PCA components provided in the dataset, additional features were engineered to improve model performance:

- **ECEF distance**: Calculated the distance of each transaction from the origin in the PCA space using Earth-Centered Earth-Fixed coordinate conversion, giving a single metric of how "unusual" a transaction is
- **Rolling statistics**: Rolling mean and standard deviation of transaction amounts to capture spending velocity
- **Time-based features**: Hour of day, day of week, and time since last transaction
- **Amount features**: Log-transformed amount and amount percentiles within rolling windows

### Models Compared

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|-------|----------|-----------|--------|----------|----------|
| Logistic Regression | 99.9% | 85.2% | 90.1% | 87.6% | 0.98 |
| Random Forest | 99.9% | 88.4% | 92.3% | 90.3% | 0.99 |
| XGBoost | 99.9% | 91.7% | 93.8% | 92.7% | 0.99 |
| Neural Network | 99.9% | 89.1% | 91.5% | 90.3% | 0.98 |

XGBoost achieved the best overall performance with strong precision and recall, making it the recommended model for production deployment.

### Statistical Hypothesis Testing

Two hypothesis tests were conducted to validate findings:

1. **Chi-Square Test**: Tested whether the distribution of transaction amounts differs significantly between fraudulent and legitimate transactions. The test confirmed a statistically significant difference, meaning fraudsters do follow a distinct amount pattern.

2. **Two-Sample t-Test**: Compared the mean ECEF distance scores between fraud and non-fraud classes. The t-test showed a significant difference, confirming that the engineered distance feature has strong discriminative power.

3. ## Phase 3 - Fraud Risk Routing System

The routing system takes the model outputs and applies a tiered decision framework to determine what action should be taken on each flagged transaction. Instead of a simple binary flag, transactions are scored and routed based on risk severity.

### Routing Logic

| Risk Tier | Score Range | Action |
|-----------|-------------|----------|
| Tier 1 - Block | 90-100 | Auto-block transaction, freeze card, notify customer immediately |
| Tier 2 - Challenge | 70-89 | Trigger 2FA verification, hold transaction pending user response |
| Tier 3 - Flag | 50-69 | Allow but flag for review within 24 hours |
| Tier 4 - Monitor | 0-49 | Allow and monitor, log for pattern analysis |

The routing decisions factor in the model probability score, transaction amount, merchant risk level, and time-of-day risk. This tiered approach reduces false positives on borderline cases while still catching high-confidence fraud.

## File Structure

```
credit-card-fraud-detection/
|-- README.md                     # This file
|-- credit_card_fraud.ipynb       # Main Jupyter notebook with full analysis
|-- report.pdf                    # Full written report
|-- reports/
|   |-- Fraud_Analysis_Dashboard.twbx  # Tableau packaged workbook
|-- data/
|   |-- creditcard.csv            # Dataset (download from Kaggle)
|-- requirements.txt              # Python dependencies
|-- .gitignore                    # Git ignore rules
```

## How to Run

1. Clone this repository:
   ```
   git clone https://github.com/DIGEN01/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it in the `data/` folder as `creditcard.csv`

4. Open and run the notebook:
   ```
   jupyter notebook credit_card_fraud.ipynb
   ```

5. View the Tableau dashboard by opening `reports/Fraud_Analysis_Dashboard.twbx` in Tableau Desktop or Tableau Reader (free download available from Tableau)

## Tools and Libraries

- **Python 3.x** - Core language for data processing and ML
- **pandas, numpy** - Data manipulation
- **scikit-learn** - Machine learning models and evaluation
- **xgboost** - Gradient boosting model
- **tensorflow / keras** - Neural network model
- **scipy, statsmodels** - Statistical hypothesis testing
- **matplotlib, seaborn** - Visualizations
- **Tableau** - Interactive dashboard and data exploration
- **Jupyter Notebook** - Interactive analysis environment

## Limitations

- The dataset contains PCA-transformed features, so the original variables (merchant type, cardholder demographics, etc.) are not available for deeper behavioral analysis
- The model was trained on 2013 European data and may not generalize well to current fraud patterns or other geographic regions
- No external features were used (e.g., IP address, device fingerprinting, geo-location APIs) that would be available in a real production system
- The routing system is rule-based and would benefit from a reinforcement learning approach in a live deployment

## License

This project is for educational purposes. The dataset is subject to Kaggle's terms of use.
