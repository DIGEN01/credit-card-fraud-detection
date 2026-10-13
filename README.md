# Credit Card Fraud Detection

![Final pipeline overview](Final%20Pipeline%20Overview.png)

An end-to-end fraud detection project on 1,296,675 simulated card transactions: Tableau for pattern discovery, three machine learning models in Python, and a probability-band routing system that turns a model score into an actual business decision.

The headline result on a held-out, strictly later time period: **96.90% precision at 89.34% recall**, catching 1,374 of 1,538 fraud cases while raising only 44 false alarms across 257,797 legitimate transactions.

The more interesting part of this project is what had to be fixed to get an honest number. An earlier version of this analysis scored considerably better and was wrong, because five of its features leaked information from the future and the train/test split let the model learn from transactions that happened after the ones it was tested on. Both are documented below, along with three design ideas that were tested and rejected.

---

## Results at a glance

Test set: 259,335 transactions from the last 20% of the timeline, containing 1,538 fraud cases.

| Model | Precision | Recall | F1 | AUC-ROC | Fraud caught | Fraud missed | Missed fraud $ | False alarms |
|---|---|---|---|---|---|---|---|---|
| ANN (threshold 0.80) | 0.9460 | 0.8544 | 0.8978 | 0.9987 | 1,314 | 224 | $56,497.79 | 75 |
| Random Forest (0.50) | 0.8864 | 0.8375 | 0.8613 | 0.9975 | 1,288 | 250 | $51,959.58 | 165 |
| XGBoost (0.50) | 0.9481 | 0.8908 | 0.9185 | 0.9994 | 1,370 | 168 | $33,174.27 | 75 |
| **Routing pipeline** | **0.9690** | **0.8934** | **0.9296** | n/a | **1,374** | **164** | **$30,511.66** | **44** |

Missed fraud is reported in dollars as well as counts, because a model that misses four small frauds is not equivalent to one that misses four large ones. Threshold choices in this project were made against the dollar column, not just F1.

---

## Dataset

| Property | Value |
|---|---|
| Source | [Kaggle: Credit Card Transactions Dataset](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset) |
| Transactions | 1,296,675 |
| Fraud cases | 7,506 (0.5789%) |
| Legitimate | 1,289,169 (99.42%) |
| Unique cards | 983 |
| Date span | 537 days |
| Target | `is_fraud` (0 = legitimate, 1 = fraud) |

With 99.42% of rows legitimate, a model that predicts "never fraud" scores 99.42% accuracy and is completely useless. Accuracy is not reported anywhere in this project. Precision, recall and missed-fraud dollars are.

The raw CSV is not in this repo because of its size. Download it from the Kaggle link and place it in `data/` as `credit_card_transactions.csv`.

---

## Phase 1: Descriptive analytics (Tableau)

Data was cleaned in Tableau Prep Builder, then explored in Tableau Desktop to establish when, where and to whom fraud happens. Each finding below became an engineered feature in Phase 2.

- **Time of day.** Fraud sits near baseline through the day and rises steeply from 21:00, staying elevated until roughly 04:00. The `is_night` feature is defined as 21:00 to 03:59 to match this observed window rather than a generic "night" guess.
- **Seasonality.** Fraud peaks in December. Online shopping fraud in that month averages roughly $1,010 per transaction, well above its already elevated baseline.
- **Category concentration.** Fraud is not spread evenly. Five categories account for roughly 78% of all fraud cases, led by `shopping_net` at 22.82% with an average fraud amount around $999.
- **Demographics.** Cardholders aged 65 and over absorb 27.36% of all fraud, a disproportionate share.
- **Amount.** Fraudulent transactions are markedly larger than a cardholder's own typical spend, which motivated the amount-ratio and amount-deviation features.

---

## Phase 2: Predictive analytics

### Leak-safe feature engineering

This is the part worth reading. Five behavioural features were originally built with a plain group-by across the whole dataset:

```python
# WRONG: every row's "average spend" includes transactions that happen after it
df["avg_amt_per_card"] = df.groupby("cc_num")["amt"].transform("mean")
```

A model trained on that knows the future. It scores beautifully and would collapse in production. All five were rebuilt to look strictly backwards:

```python
# RIGHT: a 150-day trailing window, excluding the current transaction
_rolling_sum = _time_indexed.groupby("cc_num")["amt"].rolling("150D").sum()
_rolling_count = _time_indexed.groupby("cc_num")["amt"].rolling("150D").count()
```

Counts (`tx_count_per_card`, `merch_tx_count`, `prior_fraud_count`) were rebuilt the same way, as expanding prior counts. This cost real measurable performance. That drop is the honest number.

### Chronological split

The split was also wrong. A random stratified split lets training rows sit *after* test rows on the timeline, so even leak-safe features get evaluated optimistically. It was replaced with a chronological cut at the 80th percentile of the timestamp:

- Cutoff: **2020-03-06**
- Train: 1,037,340 rows (0.5753% fraud)
- Test: 259,335 rows (0.5931% fraud)

The model is therefore trained on the past and tested on the future, which is how it would actually be used.

### Model inputs

Sixteen numeric features plus two categoricals:

`amt`, `age`, `distance`, `hour`, `week`, `month`, `is_night`, `merch_lat`, `merch_long`, `avg_amt_per_card`, `amt_ratio`, `amt_deviation`, `tx_count_per_card`, `merch_tx_count`, `prior_fraud_count`, `implied_speed_kmh`, plus `category` and `gender`.

`distance` is computed with an ECEF (Earth-Centered, Earth-Fixed) coordinate conversion rather than naive latitude/longitude subtraction, so it is a true great-circle distance. `implied_speed_kmh` divides that distance by the gap since the card's previous transaction, flagging physically impossible travel.

The ANN one-hot encodes the categoricals; the tree models use label encoding. Class imbalance is handled with class weights, not SMOTE.

### Overfitting check

| Model | Train AUC | Test AUC | AUC gap | Recall gap |
|---|---|---|---|---|
| ANN | 0.9996 | 0.9987 | 0.0009 | 0.0406 |
| Random Forest | 0.9999 | 0.9975 | 0.0024 | 0.1295 |
| XGBoost | 1.0000 | 0.9994 | 0.0006 | 0.1037 |

AUC gaps are negligible. The recall gaps on the tree models are the normal cost of a chronological split, where the test period contains fraud patterns the training period never showed.

### Statistical validation

**Cochran's Q** (omnibus, all three models at once): Q = 123.1360, df = 2, p < 0.001. The three models do not have the same error rate.

**McNemar's test** (pairwise, run only because the omnibus test was significant, which keeps the multiple-comparisons problem under control):

| Pair | p-value | Result |
|---|---|---|
| ANN vs Random Forest | 0.000000 | Significant |
| ANN vs XGBoost | 0.000066 | Significant |
| Random Forest vs XGBoost | 0.000000 | Significant |

All three pairs differ significantly, so the models genuinely make different mistakes. That is the precondition for any ensemble being worth trying.

**Mann-Whitney U** (are fraud and legitimate score distributions separable?):

| Model | Fraud mean | Legit mean | Effect size r |
|---|---|---|---|
| ANN | 0.8890 | 0.0021 | 0.9974 |
| Random Forest | 0.7848 | 0.0097 | 0.9951 |
| XGBoost | 0.8783 | 0.0010 | 0.9988 |

Pick one fraud case and one legitimate transaction at random and the model ranks the fraud higher more than 99.5% of the time. That is what makes probability-band routing defensible.

---

## Phase 3: Prescriptive analytics

Every transaction is scored once by XGBoost, then routed by how trustworthy that score is in its particular range.

| XGBoost probability | Action | How the decision is made | Transactions | Fraud | Fraud rate |
|---|---|---|---|---|---|
| Below 0.30 | AUTO_ALLOW | XGBoost alone, no further scoring | 257,767 | 133 | 0.05% |
| 0.30 to 0.40 | AI_COUNCIL_A | Weighted blend 70/15/15, cutoff 0.50 | 69 | 18 | 26.09% |
| 0.40 to 0.55 | AI_COUNCIL_B | Weighted blend 60/20/20, cutoff 0.50 | 68 | 24 | 35.29% |
| 0.55 to 0.72 | STEP_UP_AUTH | Cardholder verifies via OTP / 2FA | 81 | 44 | 54.32% |
| 0.72 and above | BLOCK_AND_REVIEW | Blocked, unless the amount exceeds $2,000 | 1,350 | 1,319 | 97.70% |

The band edges are not arbitrary. They come from a calibration check comparing predicted against observed fraud rate per probability band, which identifies exactly where XGBoost's own confidence stops tracking reality and a second opinion becomes worth paying for.

**99.4% of all transactions clear automatically** with no customer friction at all.

### What the pipeline actually buys, decomposed

The pipeline avoids 31 false positives relative to XGBoost alone. Where they come from matters:

| Band | XGBoost alone | Pipeline | FP avoided |
|---|---|---|---|
| AI_COUNCIL_B | 7 | 13 | **-6** |
| STEP_UP_AUTH | 37 | 0 | **+37** |
| BLOCK_AND_REVIEW | 31 | 31 | 0 |

The two ensemble bands are **net negative** by six false positives. Essentially the entire gain comes from the step-up tier, which resolves against the transaction's true outcome, modelling the fact that a genuine cardholder passes a 2FA challenge and a thief generally does not.

That assumption is real capability, not a trick, but it is doing almost all of the work and this project says so plainly rather than presenting the result as a clean three-model win. The notebook computes this decomposition explicitly.

---

## Three ideas that were tested and rejected

Kept in the write-up because negative results are results.

1. **Blanket ensembling.** Averaging all three models on every transaction does not beat XGBoost alone. Simple and F1-weighted averaging both land at F1 0.9159 and 0.9156 against XGBoost's 0.9185, and majority voting is worse still at 0.9084.
2. **Anomaly-type routing.** The original design routed each transaction to whichever model specialised in its anomaly type. XGBoost wins every single category, so there is nothing to route on. The idea became a hypothesis that was tested and rejected rather than a feature.
3. **Small-amount auto-allow.** Auto-approving everything under a dollar floor exploded missed fraud for almost no false-positive gain.

A related finding sets the ceiling on the whole routing idea: of the 168 fraud cases XGBoost misses, 136 are missed by all three models. **Only 32 are cases another model actually catches.** That is the real size of the opportunity, not the 171 an earlier version of this analysis claimed.

---

## Known limitations

- **983 unique cards.** Per-card behavioural history is far richer here than it would be in a real portfolio, so the behavioural features are likely optimistic.
- **Simulated data.** Fraud patterns are cleaner and more learnable than real card fraud.
- **The 2FA assumption.** Quantified above. A thief who has also compromised the cardholder's phone defeats it, and the model cannot see that.

---

## Repository structure

```
credit-card-fraud-detection/
|-- README.md
|-- Jupyter Notebook.ipynb                          # Full pipeline, all three phases
|-- Final Pipeline Overview.png                     # Routing diagram
|-- Tableau_dashboard.pdf                           # Phase 1 dashboard export
|-- reports/
|   |-- CreditCard_Fraud_Detection_Report.docx      # Full written report
|-- data/
|   |-- Dataset link.txt                            # Download instructions
|-- requirements.txt
|-- .gitignore
```

## How to run

```bash
git clone https://github.com/DIGEN01/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset), place it in `data/` as `credit_card_transactions.csv`, then:

```bash
jupyter notebook "Jupyter Notebook.ipynb"
```

XGBoost is not bitwise reproducible across runs, so the figures above may shift by a few cases on a fresh execution.

## Tools and libraries

**Python 3** with pandas, numpy, scikit-learn, xgboost, tensorflow/keras, scipy, statsmodels, matplotlib and seaborn. **Tableau Prep Builder** for cleaning and **Tableau Desktop** for the Phase 1 dashboard.

## Author

Digen Naik, Post-Baccalaureate Diploma in Business Analytics, Cape Breton University.

## License

Educational project. The dataset is subject to Kaggle's terms of use.
