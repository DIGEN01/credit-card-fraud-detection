# Credit Card Fraud Detection

A three-phase credit card fraud detection project combining descriptive analytics, Python machine learning, and a prescriptive fraud risk routing system. Built on approximately 1.8 million simulated credit card transactions with 7,506 confirmed fraud cases (0.58% fraud rate).

## Project Overview

Credit card fraud is one of the most damaging and technically complex problems in financial services. The core challenge is not just building a model that detects fraud, but building a system that is statistically rigorous and operationally deployable. Fraud detection is uniquely difficult because the data is severely imbalanced (roughly 170 legitimate transactions for every fraud case), fraud patterns shift across time and merchant category, and the cost of a missed fraud far exceeds the cost of a false alarm.

This project was built around the CRISP-DM methodology and split into three phases:

1. **Phase 1 - Descriptive Analytics**: Tableau-based pattern discovery uncovering when, where, and how fraud occurs. This phase is embedded in the Jupyter notebook alongside the ML pipeline.
2. **Phase 2 - Predictive Analytics**: Three machine learning models (ANN, Random Forest, XGBoost) with class-imbalance handling, threshold optimization, and statistical hypothesis testing.
3. **Phase 3 - Prescriptive Analytics**: A 4-tier routing system that converts model probability scores into automated business decisions.

## Dataset

| Property | Details |
|----------|----------|
| Source | [Kaggle - Credit Card Transactions Dataset](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset) |
| Total Transactions | ~1.8 million records |
| Fraud Cases | 7,506 (0.58%) |
| Legitimate | 1,289,169 (99.42%) |
| Fraud Amount (avg) | $531.30 per transaction |
| Key Numerical Features | amt, age, distance, merchlat, merchlong |
| Key Categorical Features | category, gender, merchant, state |
| Target Variable | isfraud (0 = legitimate, 1 = fraudulent) |

> Note: The raw dataset file is not included in this repo due to size. Download it from the Kaggle link above and place it in the `data/` folder as `credit_card_transactions.csv` before running the notebook.
>
> ## Phase 1 - Descriptive Analytics

The descriptive analytics phase was conducted in Tableau Desktop after data preparation in Tableau Prep. The cleaned dataset was loaded into Tableau to build a comprehensive fraud analytics dashboard that answered the foundational question: when, where, who, and how does fraud occur. All insights from this phase were then incorporated into the Jupyter notebook for the ML pipeline.

### Key Findings from Descriptive Analysis

- **Monthly Pattern**: Fraud peaks sharply in December, driven by holiday shopping. Online shopping categories average $1,010 per fraudulent transaction during this period.
- **Hourly Pattern**: Fraud concentrates between 10 PM and 3 AM, exceeding 1,200 transactions per hour at peak, which is over 10 times the daytime baseline.
- **Weekly Pattern**: Saturday, Sunday, and Monday consistently record the highest fraud counts, roughly 20-25% above the weekly average.
- **Category Concentration**: Four categories account for over 65% of fraud. Grocery POS fraud almost exclusively occurs between midnight and 3 AM, consistent with card-present skimming at unattended terminals.
- **Demographic Pattern**: Elderly cardholders (65+) bear a disproportionate share of fraud at 27.36%, while young adults (18-25) show the lowest exposure at 2.33%.
- **Amount Signal**: Average fraudulent transaction ($531.30) is substantially larger than legitimate ones ($70.35), which motivated the `amtratio` engineered feature.

These insights directly informed six of the fourteen engineered features used in the machine learning phase.

## Phase 2 - Predictive Analytics

The Jupyter notebook (`credit_card_transactions.ipynb`) contains the full ML pipeline including data preprocessing, feature engineering, model training, evaluation, and statistical testing.

### Data Preparation

All data cleaning was performed in Tableau Prep Builder before modeling, including column removal, age calculation from date of birth, null handling, and duplicate removal. The cleaned dataset was then imported into Python for additional preprocessing steps.

### Feature Engineering

Fourteen domain-specific features were engineered, with six directly motivated by the descriptive analytics phase:

- **amtratio**: Ratio of transaction amount to cardholder's historical average. A high ratio signals an unusually large purchase.
- **amtdeviation**: Standard deviation-normalized deviation from per-card mean, capturing statistical outliers.
- **avgamtpercard**: Rolling average transaction amount per card, establishing a behavioral baseline.
- **txcountpercard**: Number of transactions per card, a frequency-based anomaly signal.
- **merchtxcount**: Transaction volume per merchant, flagging high-fraud-concentration merchants.
- **isnight**: Binary flag for transactions between 22:00-06:00. Night-time transactions carry higher fraud risk.
- **hour, month, weekday**: Temporal decomposition of transaction timestamps.
- **age**: Cardholder age derived from date of birth.
- **categoryenc, genderenc**: Label-encoded categorical variables.
- **distance**: Geographic distance computed using ECEF (Earth-Centered Earth-Fixed) coordinate conversion, which accurately calculates distance on a sphere rather than naive lat/long subtraction.

### Models Trained

Three complementary algorithms were trained on an 80/20 stratified split with class-weighted handling (no SMOTE oversampling) and threshold optimization via Precision-Recall curve analysis.

| Model | AUC-ROC | Precision | Recall | F1 Score | Fraud Caught | False Alarms |
|-------|---------|-----------|--------|----------|--------------|---------------|
| ANN | 0.9986 | 0.88 | 0.84 | 0.86 | 1,275 | 153 |
| Random Forest | 0.9982 | 0.92 | 0.82 | 0.86 | 1,227 | 110 |
| XGBoost | 0.9988 | 0.93 | 0.83 | 0.88 | 1,243 | 93 |

XGBoost achieved the best overall performance with the highest AUC-ROC (0.9988), precision (0.93), and F1 score (0.88). ANN caught the most fraud cases (1,275) due to its higher recall. Random Forest produced the fewest false alarms (110).

### Feature Importance Insights

- **Random Forest**: Concentrates 61.4% of decision weight on amount features (amt, amtdeviation, amtratio), making it the amount anomaly specialist.
- **XGBoost**: Uniquely captures temporal and geographic signals (month and distance both at 12% importance) that Random Forest largely ignores.
- **ANN**: Uses One-Hot Encoding for categories, giving it independent fraud weights per merchant category, making it the strongest categorical anomaly detector.

### Statistical Hypothesis Testing

Two statistical tests were conducted to validate model performance and justify the routing ensemble design.

#### Test 1: McNemar's Test (Pairwise Model Comparison)

McNemar's Test compares two classifiers evaluated on the same test set, focusing on transactions where they disagree. A significant result (p < 0.05) means the models are making errors on genuinely different transactions, justifying their combined use in an ensemble.

| Model Pair | Both Correct | A right / B wrong | A wrong / B right | Both Wrong | p-value | Decision |
|------------|-------------|-------------------|-------------------|------------|---------|----------|
| ANN vs RF | 258,822 | 78 | 115 | 320 | 0.009560 | Reject H0 |
| ANN vs XGBoost | 258,813 | 87 | 151 | 284 | 0.000044 | Reject H0 |
| RF vs XGBoost | 258,875 | 62 | 89 | 309 | 0.034358 | Reject H0 |

All three pairwise comparisons reject the null hypothesis. Each model makes errors on a statistically distinct set of transactions. This provides formal justification for routing transactions to the domain-specialist model rather than deploying a single model.

#### Test 2: Mann-Whitney U Test (Fraud vs Legitimate Score Distributions)

The Mann-Whitney U Test compares fraud probability score distributions between confirmed fraud and legitimate transactions. It confirms that model probability outputs are meaningful and well-separated enough to support threshold-based routing.

| Model | Fraud Mean Score | Legit Mean Score | p-value | Effect Size (r) | Decision |
|-------|-----------------|------------------|---------|-----------------|----------|
| ANN | 0.8360 | 0.0032 | 0.00 | 0.9957 | Reject H0 |
| Random Forest | 0.8138 | 0.0055 | 0.00 | 0.9954 | Reject H0 |
| XGBoost | 0.8357 | 0.0039 | 0.00 | 0.9961 | Reject H0 |

All three models achieve effect sizes above 0.995, meaning if you randomly pick one fraud and one legitimate transaction, the model will assign a higher fraud probability to the fraud case in over 99.5% of instances. This near-perfect separation validates that the probability thresholds used in the routing system are statistically defensible.

## Phase 3 - Prescriptive Analytics

The prescriptive phase converts model probability scores into deterministic, automated business decisions through a 4-tier action framework. Every incoming transaction is scored by the domain-specialist model and routed based on its composite fraud probability and risk score.

### 4-Tier Routing Framework

| Tier | Condition | Action |
|------|-----------|----------|
| BLOCK + REVIEW | p >= 0.85 or score >= 70 | Block transaction, open fraud investigation case |
| ML Council | prob >= 60 and (amt > 500 or score >= 50) | All three models vote, majority wins |
| STEP-UP AUTH | prob >= 30 or score >= 20 | Prompt OTP / 2FA verification |
| AUTO ALLOW | p < 0.30 and score < 20 | Process immediately, no customer friction |

The framework was evaluated on the full held-out test set. Key results:
- **BLOCK+REVIEW precision**: 95.2% - 96.9% across all three models (when the system recommends a block, it is correct 19 times out of 20)
- **Frictionless rate**: 99.93% of legitimate transactions flow through AUTO ALLOW without any customer friction
- **Financial protection**: $547K - $670K in fraud prevented per cycle

The routing architecture concentrates genuine fraud in the highest-risk tiers while keeping the vast majority of legitimate transactions in the frictionless AUTO ALLOW tier.

## File Structure

```
credit-card-fraud-detection/
|-- README.md                          # This file
|-- credit_card_transactions.ipynb     # Main Jupyter notebook (all 3 phases)
|-- report.pdf                         # Full written report
|-- reports/                           # Tableau dashboard and visual outputs
|-- data/                              # Dataset folder
|   |-- credit_card_transactions.csv   # Download from Kaggle
|-- requirements.txt                   # Python dependencies
|-- .gitignore                         # Git ignore rules
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

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset) and place it in the `data/` folder as `credit_card_transactions.csv`

4. Open and run the notebook:
   ```
   jupyter notebook credit_card_transactions.ipynb
   ```

## Tools and Libraries

- **Python 3.x** - Core language for data processing and ML
- **pandas, numpy** - Data manipulation
- **scikit-learn** - Machine learning models and evaluation
- **xgboost** - Gradient boosting model
- **tensorflow / keras** - Neural network (ANN) model
- **scipy, statsmodels** - Statistical hypothesis testing (McNemar's Test, Mann-Whitney U)
- **matplotlib, seaborn** - Visualizations
- **Tableau Prep** - Data cleaning and preparation
- **Tableau Desktop** - Descriptive analytics dashboard
- **Jupyter Notebook** - Interactive analysis environment

## License

This project is for educational purposes. The dataset is subject to Kaggle's terms of use.
