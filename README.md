# Medical Insurance Charges — Statistical Modeling & Interactive Dashboard

M.Sc. Data Science — Semester 1
Course: Statistical Modeling with Python
Assignment: Lab-4 — Applied Statistical Modeling & Interactive Web Dashboard
Name - Kumari Shreya
Student ID - 202618041

An end-to-end data science project that combines rigorous statistical inference with an interactive Streamlit dashboard. It performs exploratory analysis, parametric and non-parametric hypothesis testing, Ordinary Least Squares regression with Gauss-Markov diagnostics, and delivers the results through a live web application.

---

## Live Dashboard

Deploy link: *(add your Streamlit Community Cloud URL here after deployment)*

---

## Repository Structure

```
insurance-dashboard/
├── data/
│   └── insurance.csv          # Medical insurance dataset
├── analysis.ipynb             # EDA, hypothesis tests, regression modeling
├── app.py                     # Streamlit dashboard (3 tabs)
├── requirements.txt           # Python dependencies
├── README.md
└── .gitignore
```

---

## Dataset

**Source:** Medical Cost Personal Dataset (Kaggle)
**Rows:** 1,338 (1,337 after dropping 1 duplicate)
**Columns:** 7

| Column | Type | Description |
|---|---|---|
| age | int | Age of primary beneficiary |
| sex | categorical | female / male |
| bmi | float | Body mass index |
| children | int | Number of dependents covered |
| smoker | categorical | yes / no |
| region | categorical | northeast / northwest / southeast / southwest |
| charges | float | Individual medical costs billed by health insurance |

**Target variable:** `charges`

**Dataset characteristics:**
- No missing values
- 1 duplicate row removed
- `charges` is heavily right-skewed (skewness = 1.52, kurtosis = 1.61)

---

## Setup & Run

### Local (recommended)

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd insurance-dashboard

# 2. Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the dashboard
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

### Google Colab (fallback)

```python
!pip install streamlit
%%writefile app.py
# ...paste app.py contents...

!streamlit run app.py & npx localtunnel --port 8501
```

---

## Statistical Findings

### Part 1 — Exploratory Data Analysis

**Descriptive statistics (numeric features):**

| Feature | Mean | Median | Std | IQR | Skewness | Kurtosis |
|---|---|---|---|---|---|---|
| age | 39.21 | 39.00 | 14.05 | 24.00 | 0.06 | −1.25 |
| bmi | 30.66 | 30.40 | 6.10 | 8.40 | 0.28 | −0.05 |
| children | 1.09 | 1.00 | 1.21 | 2.00 | 0.94 | 0.20 |
| charges | 13,270.42 | 9,382.03 | 12,110.01 | 11,899.63 | **1.52** | 1.61 |

**Key observations:**
- `charges` is heavily right-skewed and bimodal (driven by smoker/non-smoker split)
- `bmi` is approximately normal
- No multicollinearity among predictors (all pairwise |r| < 0.31)

---

### Hypothesis Test 1 — Smokers vs Non-Smokers on Charges

**H0:** There is no difference in mean charges between smokers and non-smokers
**H1:** There is a significant difference
**α = 0.05** (two-tailed)

**Assumption checks:**
- Shapiro-Wilk: both groups significantly non-normal (p = 3.63e-09 and 1.50e-28)
- Levene's test: variances significantly unequal (p = 1.67e-66)

**Test used:** Mann-Whitney U (non-parametric, because normality failed)

**Result:**
- U = 283,859.0
- p = 5.75e-130
- Smokers mean: $32,050.23 | Non-smokers mean: $8,440.66
- Difference: **+$23,609.57**

**Decision:** **Reject H0.** There is overwhelming statistical evidence that smokers incur significantly higher medical charges than non-smokers.

---

### Hypothesis Test 2 — Charges across Geographic Regions

**H0:** Average charges are equal across all 4 regions
**H1:** At least one region differs significantly
**α = 0.05**

**Assumption checks:**
- Shapiro-Wilk per region: all four regions significantly non-normal (p < 1e-17)
- Levene's test: variances significantly unequal (p = 8.69e-04)

**Tests performed:**
- One-Way ANOVA (parametric): F = 2.9261, p = 0.0328 → would reject H0
- Kruskal-Wallis (non-parametric): H = 4.6225, p = 0.2016 → fail to reject H0

**Decision:** **Fail to Reject H0**, using Kruskal-Wallis as authoritative. The ANOVA's assumptions (normality, equal variance) are violated, making its p-value unreliable. The non-parametric Kruskal-Wallis test, which makes no such assumptions, shows no significant regional difference at α = 0.05.

---

### Part 2 — Regression Modeling

**Model comparison:**

| Model | R² | Adj. R² | Residual Skew | Residual Kurtosis |
|---|---|---|---|---|
| Main effects OLS | 0.751 | 0.749 | 1.21 | 5.65 |
| Log(charges) OLS | 0.768 | 0.766 | 1.68 | 7.32 |
| **Interaction OLS (smoker × bmi)** | **0.841** | **0.840** | 2.53 | 10.37 |
| **Interaction + HC3 robust SEs** | **0.841** | **0.840** | 2.53 | 10.37 |

**Final model:**

```
charges = β0 + β1·age + β2·bmi + β3·children + β4·smoker_yes
        + β5·sex_male + β6·region_NW + β7·region_SE + β8·region_SW
        + β9·(smoker_yes × bmi) + ε
```

**Fitted with HC3 robust standard errors** to correct for heteroscedasticity.

**Coefficients (HC3 robust):**

| Predictor | Coef ($) | p-value | Significant |
|---|---|---|---|
| const | −2,222.34 | 0.004 | Yes |
| age | +263.56 | < 0.001 | Yes |
| bmi | +23.58 | 0.297 | No |
| children | +515.96 | < 0.001 | Yes |
| smoker_yes | −20,420 | < 0.001 | Yes (intercept shift) |
| sex_male | −498.94 | 0.062 | No |
| region_northwest | −583.03 | 0.151 | No |
| region_southeast | −1,210.29 | 0.003 | Yes |
| region_southwest | −1,231.12 | 0.001 | Yes |
| **smoker_bmi** | **+1,443.07** | **< 0.001** | **Yes** |

**Key insight — the smoker × bmi interaction:**

Because the interaction term is included, the effect of BMI depends on smoking status:
- Non-smokers: +$23.58 per BMI unit (not significant, p = 0.297)
- Smokers: +$1,466.65 per BMI unit (p < 0.001)

BMI barely affects costs for non-smokers, but dramatically increases costs for smokers. The interaction is the dominant effect in the model.

**Gauss-Markov diagnostics:**

| Assumption | Result |
|---|---|
| Linearity | Questionable — residuals show curvature |
| Homoscedasticity | Violated — corrected via HC3 robust SEs |
| Normality of residuals | Violated (JB p < 0.001) — expected for this dataset |
| No autocorrelation | Passed (Durbin-Watson ≈ 2.05) |
| No multicollinearity | Passed (all VIF < 2) |

**Note on residual non-normality:** The insurance dataset has a bimodal, smoker-driven distribution with heavy tails. Residual non-normality is a known characteristic. At n = 1337, the Central Limit Theorem ensures that the coefficient estimates remain asymptotically valid, and the HC3 robust standard errors address the heteroscedasticity.

---

## Dashboard Features

The Streamlit dashboard (`app.py`) has three tabs:

### Tab 1 — Data Exploration
- Sidebar filters: age range slider, BMI range slider, smoker multi-select, region multi-select
- Live-updating summary statistics table
- Distribution of charges (histogram + box)
- Age vs charges scatter, colored by smoker status
- Charges by region boxplot

### Tab 2 — Hypothesis Testing Lab
- Dropdown menus to select any categorical variable and any numeric metric
- Adjustable significance level (α) slider
- Automatic Shapiro-Wilk normality check per group (with results table)
- Automatic Levene's test for equal variance
- Auto-selection of the appropriate test:
  - 2 groups → t-test (Student or Welch) or Mann-Whitney U
  - 3+ groups → One-Way ANOVA or Kruskal-Wallis
- Clear Reject / Fail to Reject H0 verdict
- Group summary statistics

### Tab 3 — Live Prediction & Diagnostics
- Interactive sliders and radio buttons for all model inputs
- Real-time prediction of medical charges
- 95% confidence interval for the mean response
- 95% prediction interval for an individual (clamped at 0)
- Coefficient table with HC3 robust standard errors
- Model performance metrics: R², Adjusted R², skew, kurtosis
- Residual diagnostics: Q-Q plot and Residuals-vs-Fitted plot
- Omnibus, Jarque-Bera, and Durbin-Watson statistics

---

## Technical Stack

- **Python 3.10+**
- **pandas, numpy** — data manipulation
- **scipy, statsmodels** — statistical tests and OLS regression
- **matplotlib, seaborn, plotly** — visualization
- **streamlit** — interactive web dashboard

---

## Limitations & Notes

1. **Residual non-normality:** Inherent to the dataset's bimodal structure. Addressed through HC3 robust standard errors and the large sample size (CLT).
2. **Interaction interpretation:** When interactions are present, individual main-effect coefficients (e.g., `smoker_yes`, `bmi`) cannot be interpreted in isolation — they must be interpreted jointly.
3. **Prediction interval bound:** Linear models can produce negative lower bounds for strictly positive targets. Clamped to 0 in the dashboard.

---
