# Predictive Churn & Retention Analysis Platform

An intelligent, interactive **Predictive Churn & Customer Retention Analysis Platform** designed for subscription-based businesses (SaaS, Telecom, Streaming, and Content Providers). The system leverages machine learning to ingest customer behavioral metrics, perform robust cleaning and feature engineering, fit **Logistic Regression** classification models for churn risk scoring, and project financial impact using **Linear Regression** time-series forecasting.

---

## 🚀 Key Features & Interactive Dashboards

The application is structured into **seven dedicated tabs** representing an end-to-end data science and business intelligence workflow:

1. **📂 Dataset Overview**: Upload custom datasets (CSV, Excel, or JSON) or load a high-fidelity **Simulated Customer Dataset** featuring realistic statistical correlations, custom missing-value rates, duplicate records, and explicit outliers to instantly test the platform.
2. **🧹 Data Cleaning & Validation**: 
   - Handles missing data (Mean, Median, Mode Imputation, or Row Removal).
   - Detects and filters outliers dynamically using **IQR** (Interquartile Range) or **Z-Score** thresholds.
   - Cleans duplicate records and conflicting Customer IDs.
   - Displays a **Data Validation Dashboard** showing records processed and comparing distribution metrics.
3. **📊 Behavioral Analytics**: Explores trends through interactive Plotly visualizations (Renewed vs. Churned breakdown, login frequencies, session duration distributions, plan-type renewal rates, customer health scores, and tenure cohorts).
4. **🤖 Churn Prediction (Logistic Regression)**: 
   - Trains a Logistic Regression model with custom parameters (train-test split, L1/L2 penalties, and C regularization strength).
   - Generates key classification metrics: **Accuracy, Precision, Recall, F1 Score, and ROC-AUC**.
   - Displays dynamic **ROC Curves**, **Confusion Matrices**, and statistical model summary tables (co-efficient tables with standard errors, odds-ratios, and p-values).
5. **⚠️ Risk Assessment Scorecard**: 
   - Ranks and sorts the entire customer database by calculated probability.
   - Segments users into actionable risk tiers: **High Churn Risk 🔴 (0-30%)**, **Medium Risk 🟡 (31-70%)**, and **Likely Renewal 🟢 (71-100%)**.
   - **Interactive Individual Prediction Tool**: Slide/input custom indicators (Logins, tenure, complaints) to calculate renewal probability, view a real-time risk scorecard, and pull programmatic business retention actions.
6. **📈 Financial Forecast & Churn Revenue Impact**: 
   - Details expected renewal volumes and Monthly Recurring Revenue (MRR) retained vs. at risk.
   - Fits a **Linear Regression** time-series trend model on historical customer cohorts to forecast the next 6 months of churn rates and revenue.
7. **📄 Executive Business Reports**:
   - Generates and downloads a print-ready **PDF Executive Summary Report** using ReportLab.
   - Exports the fully scored customer database as a styled **Microsoft Excel Spreadsheet** using OpenPyXL, including colored risk highlights and conditional column layouts.

---

## 📐 Mathematical Formulation

### 1. Renewal Probability (Logistic Regression)
The likelihood of a customer renewing their subscription is modeled as:

$$P(\text{Renewal}) = \sigma(\beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_n X_n)$$

Where $\sigma(z)$ is the sigmoid function:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

- **Independent Variables ($X_i$)**: Tenure, Monthly Spend, Login Frequency, Session Duration, Support Tickets, Product Usage, Last Login Days, Plan Type, and engineered scores.
- **Log-Odds Impact**: Coefficients ($\beta_i$) represent the impact on renewal. The **Odds Ratio ($e^{\beta_i}$)** translates coefficients into multiplicative factor changes in the odds of renewal.

### 2. Time-Series Trends (Linear Regression)
Monthly cohort renewal rates and total revenue trends are projected forward using:

$$\hat{y}_t = \alpha + \gamma t$$

Where:
- $t$ is the month index (historical to forecasted).
- $\hat{y}_t$ is the predicted renewal rate or revenue.
- $\gamma$ represents the monthly linear growth/decline rate (slope).

---

## 🛠️ Technology Stack
- **Dashboard Interface**: [Streamlit](https://streamlit.io/)
- **Data Wrangling**: Pandas, NumPy
- **Machine Learning & Stats**: Scikit-Learn, Statsmodels
- **Interactive Visualizations**: Plotly
- **Spreadsheet & Reporting**: ReportLab, OpenPyXL

---

## ⚙️ Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/abakaushik-lgtm/PredictiveChurnAnalysis.git
cd "Predictive Churn Analysis"
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv .venv
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Verification Tests
```bash
python verify_app.py
```

### 5. Launch the Web Application
```bash
streamlit run app.py
```
This opens the interactive dashboard in your default browser at `http://localhost:8501`.
