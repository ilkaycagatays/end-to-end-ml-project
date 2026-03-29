# 🚚 Logistics Operations Optimization — Data Analysis Portfolio Project

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?logo=pandas)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-1.3-F7931E?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

> **Junior Data Analyst Portfolio Project** — End-to-end analysis of 15,000 logistics deliveries across Turkey, including delay analysis, cost optimization, and workload forecasting using synthetic ERP/WMS-like data.

---

## 📋 Table of Contents

* [Project Overview](#-project-overview)
* [Business Questions & Insights](#-business-questions--insights)
* [Dataset](#-dataset)
* [Project Structure](#-project-structure)
* [Installation](#-installation)
* [How to Run](#-how-to-run)
* [Machine Learning Models](#-machine-learning-models)
* [Key Findings](#-key-findings)
* [Contact](#-contact)

---

## 🎯 Project Overview

This project analyzes operational logistics data to answer key business questions:

| Question                                | Approach            | Output                        |
| --------------------------------------- | ------------------- | ----------------------------- |
| Which regions have the highest delays?  | EDA + Visualization | City-level delay map          |
| What are the main causes of delays?     | Pareto Analysis     | Top 3 causes (80/20 rule)     |
| Can workload be forecasted?             | Time Series         | 30-day demand forecast        |
| How can workforce planning be improved? | Operational Formula | Driver requirement estimation |
| Will a delivery be delayed?             | Random Forest       | Early warning classifier      |

**Scope:**

* 📅 2023–2024 (2 years)
* 🏙️ 17 cities across Turkey
* 📦 15,000 orders
* 🚚 120 drivers
* 🏭 8 warehouses, 730 days of performance data

---

## ❓ Business Questions & Insights

### 1. Which regions experience more delays?

Eastern and central regions show **2–3x higher delay rates** compared to western cities.

```
Diyarbakir → 41.7% delay  ⚠️ Critical
Erzurum    → 37.2% delay  ⚠️ Critical
Trabzon    → 34.1% delay  ⚠️ High
Bursa      → 14.3% delay  ✅ Low
Kocaeli    → 15.1% delay  ✅ Low
```

---

### 2. What are the main causes of delays?

Pareto analysis shows that **3 factors account for ~40% of delays**:

| Cause              | Cases | Share | Avg Duration |
| ------------------ | ----- | ----- | ------------ |
| Weather Conditions | 687   | 21.0% | 2.0 days     |
| Road Construction  | 310   | 9.5%  | 2.0 days     |
| Warehouse Delay    | 303   | 9.3%  | 2.0 days     |

👉 **Recommendation:** Implement weather monitoring & alternative routing during winter months.

---

### 3. Can workload be forecasted?

Yes. Time series analysis shows:

* **Nov–Dec:** +40% demand spike (Black Friday effect)
* **Fridays:** Weekly peak (+15%)
* **Sundays:** Lowest demand (-50%)
* **Forecast accuracy:** ±15% MAE over 30 days

---

### 4. How can workforce planning be improved?

Formula:

```
Required Drivers = ceil((Forecasted Orders / 12) × 1.15)
```

👉 Peak season requires **~30% additional workforce**

---

## 📁 Dataset

Synthetic dataset simulating ERP/WMS systems.

### Tables

| File                        | Rows   | Columns | Description           |
| --------------------------- | ------ | ------- | --------------------- |
| `orders.csv`                | 15,000 | 12      | Order data            |
| `deliveries.csv`            | 15,000 | 18      | Delivery + delay info |
| `warehouse_performance.csv` | 5,848  | 15      | Warehouse metrics     |
| `driver_performance.csv`    | 120    | 14      | Driver profiles       |

---

### Key Features

```
orders.csv
- order_id
- order_date
- destination_city
- product_category
- weight_kg
- order_value_tl
- priority

deliveries.csv
- is_delayed (target)
- delay_days
- delay_reason
- customer_satisfaction
- delivery_cost_tl
```

---

### Realism Features

* Seasonality (Nov–Dec +40%)
* Population-weighted demand (Istanbul ~28%)
* Geographic delay differences
* Priority impact (Economy = 1.5x more delays)

---

## 🗂️ Project Structure

```
logistics-optimization/
│
├── data/
├── src/
├── models/
├── reports/
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/logistics-optimization.git
cd logistics-optimization

python -m venv venv
source venv/bin/activate     # macOS/Linux
venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

---

## 🚀 How to Run

```bash
python src/generate_data.py
python src/eda_analysis.py
python src/train_models.py
```

---

## 🤖 Machine Learning Models

### 1. Delay Classification

* Model: Random Forest
* ROC-AUC: 0.665
* Accuracy: 68.6%

**Use case:** Predict delay risk at order time

---

### 2. Delay Duration Prediction

* Model: Gradient Boosting
* MAE: 1.29 days

**Use case:** Estimate delay duration for customer communication

---

### 3. Demand Forecasting

* Approach: Moving Average + EWMA
* Horizon: 30 days
* Accuracy: ±15%

**Use case:** Workforce and capacity planning

---

## 📊 Key Findings

* Overall delay rate: **21.8%**
* Average delay: **2.0 days**
* Most problematic city: **Diyarbakir**
* Best performing city: **Bursa**
* Peak demand: **December (+40%)**

---

## 💡 Operational Recommendations

1. Winter stock pre-positioning (+20%)
2. Dynamic driver allocation (+15% on Fridays)
3. Early warning system (ML-based)
4. Warehouse efficiency improvements

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* Joblib

---

## 📬 Contact

* LinkedIn: https://linkedin.com/in/your-username
* GitHub: https://github.com/your-username

---

*This project is for portfolio purposes. All data is synthetic.*
