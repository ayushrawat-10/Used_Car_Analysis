# Used Car Market Analysis and Price Prediction in India

> **IBM × Bharat Cares Data Analysis Internship Project**

A complete end-to-end data science project that analyses the Indian used-car market and builds machine learning models to predict a car's selling price from its attributes.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)
- [Key Findings](#key-findings)
- [Model Results](#model-results)
- [Output Files](#output-files)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)

---

## Project Overview

This project has two major components:

**A. Used Car Market Analysis**  
Explore pricing patterns across manufacturer, fuel type, transmission, car age, mileage, number of previous owners, city, and car rating.

**B. Used Car Price Prediction**  
Train and compare four regression models to predict a car's listing price. A Streamlit web app lets you interact with the analysis and get live price estimates.

---

## Dataset

| Property | Value |
|---|---|
| File | `Used_Car_Price_Prediction.csv` |
| Source | Indian used-car marketplace (Kaggle) |
| Records | 7,400 raw → 7,396 after cleaning |
| Columns | 29 |
| Target | `sale_price` (listing price in INR) |
| Period | 2019 – 2021 (86% from 2021) |
| Cities | 13 major Indian metros |
| Manufacturers | 27 brands |

---

## Project Structure

```
DAta_analysis/
│
├── Used_Car_Price_Prediction.csv   # Raw dataset
│
├── used_car_analysis_india.py      # Complete analysis script (run top-to-bottom)
├── app.py                          # Streamlit web frontend
│
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── Used_Car_Project_Report.docx    # Full written project report
│
└── plot_01_sale_price_distribution.png   # ┐
    plot_02_year_distribution.png         # │
    plot_03_kms_distribution.png          # │
    plot_04_categorical_frequencies.png   # │
    plot_05_price_vs_year.png             # ├─ EDA plots (auto-generated)
    plot_06_price_vs_kms.png              # │
    plot_07_price_by_make.png             # │
    plot_08_price_by_fuel.png             # │
    plot_09_price_by_transmission.png     # │
    plot_10_price_by_owners.png           # │
    plot_11_price_by_city.png             # │
    plot_12_price_by_rating.png           # │
    plot_13_correlation_heatmap.png       # │
    plot_14_model_comparison.png          # │
    plot_15_actual_vs_predicted.png       # │
    plot_16_feature_importance.png        # ┘
```

---

## Setup and Installation

### Prerequisites

- Python 3.9 or higher
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**

| Package | Version |
|---|---|
| pandas | 3.0.6 |
| numpy | 2.4.6 |
| matplotlib | 3.11.2 |
| seaborn | 0.13.2 |
| scikit-learn | 1.9.1 |
| streamlit | 1.64.0 |

---

## How to Run

### Option 1 — Run the full analysis script

Runs all 16 sections in sequence: data loading, cleaning, EDA, feature engineering, leakage analysis, model training, evaluation, and final summary. Saves all 16 plots as `.png` files.

```bash
python used_car_analysis_india.py
```

### Option 2 — Launch the Streamlit web app

Provides an interactive dashboard with five pages:

| Page | What you get |
|---|---|
| 🏠 Overview | Project summary, metrics, and key insights |
| 🔍 Dataset Explorer | Browse, filter, and inspect the data |
| 📊 EDA & Market Insights | Interactive charts and insight tables |
| 🤖 Model Evaluation | Metrics, actual vs predicted, feature importance |
| 💰 Price Predictor | Enter car details → get a live price estimate |

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

> **Note:** The Streamlit app trains all models on first load (cached). Subsequent page switches are instant.

---

## Key Findings

1. **Market dominance** — Maruti (43%) and Hyundai (24%) account for over two-thirds of all listings.
2. **Depreciation** — Median price falls from ₹5.83 L (1-year-old) to ₹0.73 L (15-year-old).
3. **Transmission premium** — Automatic cars average ₹7.09 L vs ₹4.08 L for manual — a **74% premium**.
4. **Diesel pricing** — Diesel cars have higher median prices than petrol, driven by their prevalence in SUVs and larger sedans.
5. **Ownership history** — Each additional previous owner is associated with a ~₹50,000–65,000 price reduction.
6. **Geography** — Chennai has the highest median price (₹4.51 L); Kolkata the lowest (₹3.10 L).
7. **Car rating** — "Great"-rated cars (₹4.09 L median) cost ~60% more than "good"-rated cars (₹2.55 L).
8. **Mileage** — `kms_run` has a negative correlation with price (r = −0.34).

---

## Model Results

All models were evaluated on a **20% hold-out test set (1,480 samples)**.

| Model | MAE (₹) | RMSE (₹) | R² |
|---|---|---|---|
| Linear Regression | 1,27,513 | 2,17,728 | 0.4502 |
| Decision Tree | 82,402 | 1,43,647 | 0.7607 |
| Random Forest | 69,486 | 1,13,970 | 0.8494 |
| **Gradient Boosting** | **69,210** | **1,12,080** | **0.8543** ✅ |

**Best model:** Gradient Boosting — explains **85.4%** of price variance with an average error of ~₹69,000.

### Top Features (Gradient Boosting)

| Feature | Importance |
|---|---|
| body_type | 0.3679 |
| car_age | 0.2821 |
| make | 0.1499 |
| fuel_type | 0.0544 |
| kms_run | 0.0370 |

> Feature importance reflects predictive association within the model — not causal influence.

### Data Leakage Prevention

The following columns were excluded from the ML model to prevent leakage:

| Column | Reason |
|---|---|
| `broker_quote` | Correlation 0.96 with `sale_price` — derived from the same pricing process |
| `emi_starts_from` | Correlation ≈ 1.0 — calculated as a fraction of `sale_price` |
| `booking_down_pymnt` | Correlation ≈ 1.0 — same as above |
| `original_price` | 44% missing; represents showroom price, often unavailable |
| `times_viewed` | Post-listing engagement metric — unknown at prediction time |
| `is_hot` | Post-listing engagement flag — unknown at prediction time |
| `reserved` | Assigned after buyer interaction — post-listing |

---

## Output Files

| File | Description |
|---|---|
| `plot_01_sale_price_distribution.png` | Histogram of sale price (raw + log-transformed) |
| `plot_02_year_distribution.png` | Listings by year of manufacture |
| `plot_03_kms_distribution.png` | Distribution of kilometres driven |
| `plot_04_categorical_frequencies.png` | Fuel type, body type, transmission, rating counts |
| `plot_05_price_vs_year.png` | Median price by year of manufacture |
| `plot_06_price_vs_kms.png` | Sale price vs kilometres driven (scatter) |
| `plot_07_price_by_make.png` | Median price — top 12 manufacturers |
| `plot_08_price_by_fuel.png` | Price distribution by fuel type (boxplot) |
| `plot_09_price_by_transmission.png` | Price by transmission type (boxplot) |
| `plot_10_price_by_owners.png` | Price by number of previous owners (boxplot) |
| `plot_11_price_by_city.png` | Price distribution by city (boxplot) |
| `plot_12_price_by_rating.png` | Price by car rating (boxplot) |
| `plot_13_correlation_heatmap.png` | Correlation matrix — numerical features |
| `plot_14_model_comparison.png` | MAE / RMSE / R² comparison across models |
| `plot_15_actual_vs_predicted.png` | Actual vs predicted + residual plot (best model) |
| `plot_16_feature_importance.png` | Top 20 feature importances (Gradient Boosting) |

---

## Limitations

- Dataset covers only one marketplace and 13 cities — not representative of the full Indian market.
- `sale_price` is the **listing price**, not the final negotiated transaction price.
- Luxury brands (Audi, BMW, Mercedes Benz) are under-represented (<1% each).
- Data period (2019–2021) does not reflect post-2021 supply-chain and EV market shifts.
- Missing features: engine displacement, fuel efficiency (km/L), accident history, service records.

---

## Future Improvements

- Collect larger, more recent datasets (2022–2024).
- Add XGBoost, LightGBM, or CatBoost models.
- Perform systematic hyperparameter tuning (GridSearchCV / Optuna).
- Include macroeconomic indicators (fuel prices, loan interest rates).
- Add vehicle specifications: engine size, safety rating, service history.
- Deploy the model as a REST API (FastAPI) or extend the Streamlit app for cloud hosting.

---

## Acknowledgements

This project was developed as part of the **IBM × Bharat Cares Data Analysis Internship** program.

---

*Made with Python, Scikit-learn, Streamlit, and IBM Bob.*
