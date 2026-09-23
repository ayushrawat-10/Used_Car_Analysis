# =============================================================================
# USED CAR MARKET ANALYSIS AND PRICE PREDICTION IN INDIA
# IBM x Bharat Cares Data Analysis Internship Project
# =============================================================================
# Dataset : Used_Car_Price_Prediction.csv
# Target   : sale_price (selling price of a used car in INR)
# Author   : [Your Name]
# =============================================================================

# =============================================================================
# SECTION 1 - IMPORTS AND CONFIGURATION
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend - saves plots to disk
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Reproducibility seed used throughout the project
RANDOM_STATE = 42

# Reference year used to compute car_age from yr_mfr.
# The dataset ads were created mostly in 2021, so we use 2021 as the
# reference year - this avoids introducing information from outside the data.
REFERENCE_YEAR = 2021

# Matplotlib / Seaborn style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 100, "figure.facecolor": "white"})


# =============================================================================
# SECTION 2 - LOAD DATASET
# =============================================================================

df_raw = pd.read_csv("Used_Car_Price_Prediction.csv")

print("=" * 65)
print("  USED CAR MARKET ANALYSIS AND PRICE PREDICTION IN INDIA")
print("=" * 65)


# =============================================================================
# SECTION 3 - DATASET INSPECTION
# =============================================================================

print("\n--- Dataset Shape ---")
print(f"Rows: {df_raw.shape[0]:,}   Columns: {df_raw.shape[1]}")

print("\n--- First 5 Rows ---")
print(df_raw.head())

print("\n--- Column Names ---")
print(df_raw.columns.tolist())

print("\n--- Data Types ---")
print(df_raw.dtypes)

print("\n--- Missing Values ---")
missing = df_raw.isnull().sum()
missing_pct = (missing / len(df_raw) * 100).round(2)
missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
print(missing_df[missing_df["Missing Count"] > 0])

print(f"\n--- Duplicate Rows ---")
print(f"Duplicate rows found: {df_raw.duplicated().sum()}")

print("\n--- Basic Statistics (Numerical Columns) ---")
print(df_raw.describe())

print("\n--- Unique Values per Column ---")
for col in df_raw.columns:
    print(f"  {col}: {df_raw[col].nunique()} unique values")

# -----------------------------------------------------------------------
# DATASET DESCRIPTION
# -----------------------------------------------------------------------
# The dataset contains 7,400 used-car listings from an Indian online
# marketplace with 29 columns covering vehicle attributes, listing metadata,
# and derived financial fields.
#
# Key columns:
#   car_name, make, model   - vehicle identity
#   yr_mfr                  - year of manufacture
#   fuel_type               - petrol / diesel / cng / lpg / electric
#   kms_run                 - total kilometres driven
#   sale_price              - listing/selling price in INR  <- TARGET
#   city, registered_state  - geographic information
#   body_type               - hatchback / sedan / suv / luxury
#   transmission            - manual / automatic
#   total_owners            - number of previous owners
#   car_rating              - dealer's value assessment
#   broker_quote            - dealer's internal price estimate
#   original_price          - original showroom price (44% missing)
#   emi_starts_from         - derived from sale_price (LEAKAGE risk)
#   booking_down_pymnt      - derived from sale_price (LEAKAGE risk)
#   times_viewed, is_hot    - post-listing engagement data (LEAKAGE risk)
# -----------------------------------------------------------------------


# =============================================================================
# SECTION 4 - DATA CLEANING
# =============================================================================

df = df_raw.copy()

# 4.1  Remove duplicate rows
# Only 1 duplicate row was found; removing it avoids model bias.
before = len(df)
df.drop_duplicates(inplace=True)
print(f"\nDuplicates removed: {before - len(df)}")

# 4.2  Remove rows with sale_price == 0
# These are clearly data-entry errors - a car cannot have a zero price.
zero_price = (df["sale_price"] == 0).sum()
df = df[df["sale_price"] > 0].copy()
print(f"Rows with sale_price == 0 removed: {zero_price}")

# 4.3  Parse ad_created_on as a datetime column
df["ad_created_on"] = pd.to_datetime(df["ad_created_on"], errors="coerce")

# 4.4  Standardise string columns to lowercase and strip whitespace
# Exclude boolean columns - they don't have a .str accessor.
str_cols = (df.select_dtypes(include="object").columns.tolist() +
            df.select_dtypes(include="string").columns.tolist())
bool_cols = df.select_dtypes(include="bool").columns.tolist()
str_cols = [c for c in str_cols if c not in bool_cols]
for col in str_cols:
    df[col] = df[col].astype(str).str.lower().str.strip()
    # Re-cast NaN strings back to actual NaN so downstream imputers work
    df[col] = df[col].replace("nan", np.nan)

# 4.5  Fill missing categorical values with the mode
# Missing rate is low (<10%) for body_type, transmission, source,
# registered_city/state - mode imputation preserves distribution shape.
low_miss_cats = ["body_type", "transmission", "source",
                 "registered_city", "registered_state",
                 "car_availability", "car_rating"]
for col in low_miss_cats:
    if col in df.columns and df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"  {col}: missing values filled with mode ('{mode_val}')")

# fitness_certificate: boolean-like; fill missing with False (most conservative)
df["fitness_certificate"] = df["fitness_certificate"].fillna(False)

# 4.6  original_price - 44% missing, NOT used as a predictive feature
# (also flagged as leakage - see Section 7).  We retain it only for the
# EDA insight on discount from original price.

# 4.7  Outlier treatment for kms_run
# Values above 400,000 km are extremely rare and likely data errors for the
# Indian market context; we cap them at the 99th percentile.
kms_99 = df["kms_run"].quantile(0.99)
df["kms_run"] = df["kms_run"].clip(upper=kms_99)

print(f"\nDataset after cleaning: {df.shape[0]:,} rows x {df.shape[1]} columns")


# =============================================================================
# SECTION 5 - DATA TYPE AND FEATURE CLASSIFICATION
# =============================================================================

# TARGET VARIABLE
TARGET = "sale_price"

# IDENTIFIER COLUMNS - unique per listing, no predictive value
IDENTIFIERS = ["car_name", "variant", "rto", "registered_city",
               "ad_created_on", "model"]

# LEAKAGE COLUMNS - derived from / highly correlated with sale_price
# broker_quote  : correlation 0.96 with sale_price - this is the dealer's
#                 internal price estimate computed FROM the listing price.
# emi_starts_from, booking_down_pymnt : correlation ~1.0 - these are
#                 literally calculated as fractions of the sale_price.
# original_price: 44% missing; also represents information not always
#                 available prior to listing.
# times_viewed, is_hot, reserved : generated AFTER the listing is published,
#                 so unavailable at prediction time.
LEAKAGE_COLS = ["broker_quote", "emi_starts_from", "booking_down_pymnt",
                "original_price", "times_viewed", "is_hot", "reserved"]

# NUMERICAL FEATURES
NUM_FEATURES = ["yr_mfr", "kms_run", "total_owners"]

# CATEGORICAL FEATURES
CAT_FEATURES = ["fuel_type", "body_type", "transmission",
                "city", "registered_state", "car_rating",
                "make", "source", "car_availability",
                "assured_buy", "fitness_certificate", "warranty_avail"]

print("\n--- Feature Classification ---")
print(f"Target         : {TARGET}")
print(f"Numerical      : {NUM_FEATURES}")
print(f"Categorical    : {CAT_FEATURES}")
print(f"Identifiers    : {IDENTIFIERS}")
print(f"Leakage        : {LEAKAGE_COLS}")


# =============================================================================
# SECTION 6 - EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

# Helper: INR formatter for axes
def inr_fmt(x, _):
    if x >= 1e6:
        return f"Rs.{x/1e6:.1f}M"
    if x >= 1e3:
        return f"Rs.{x/1e3:.0f}K"
    return f"Rs.{x:.0f}"


# ------------------------------------------------------------------
# 6.1  Univariate - Sale Price Distribution
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Distribution of Sale Price", fontsize=15, fontweight="bold")

axes[0].hist(df[TARGET] / 1e5, bins=60, color="#3b82d4", edgecolor="white")
axes[0].set_xlabel("Sale Price (Rs. Lakhs)")
axes[0].set_ylabel("Number of Listings")
axes[0].set_title("Histogram of Sale Price")

axes[1].hist(np.log1p(df[TARGET]), bins=60, color="#7c5cd8", edgecolor="white")
axes[1].set_xlabel("log(Sale Price)")
axes[1].set_ylabel("Number of Listings")
axes[1].set_title("Log-Transformed Sale Price (more symmetric)")

plt.tight_layout()
plt.savefig("plot_01_sale_price_distribution.png")
plt.show()
print("Saved: plot_01_sale_price_distribution.png")


# ------------------------------------------------------------------
# 6.2  Univariate - Manufacturing Year Distribution
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5))
year_counts = df["yr_mfr"].value_counts().sort_index()
ax.bar(year_counts.index.astype(str), year_counts.values,
       color="#3b82d4", edgecolor="white")
ax.set_title("Listings by Year of Manufacture", fontsize=13, fontweight="bold")
ax.set_xlabel("Year of Manufacture")
ax.set_ylabel("Number of Listings")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plot_02_year_distribution.png")
plt.show()
print("Saved: plot_02_year_distribution.png")


# ------------------------------------------------------------------
# 6.3  Univariate - Kilometres Driven Distribution
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df["kms_run"] / 1000, bins=50, color="#3b82d4", edgecolor="white")
ax.set_title("Distribution of Kilometres Driven", fontsize=13, fontweight="bold")
ax.set_xlabel("Kilometres Driven (Thousands)")
ax.set_ylabel("Number of Listings")
plt.tight_layout()
plt.savefig("plot_03_kms_distribution.png")
plt.show()
print("Saved: plot_03_kms_distribution.png")


# ------------------------------------------------------------------
# 6.4  Univariate - Categorical Columns Frequency
# ------------------------------------------------------------------
cat_plot_cols = ["fuel_type", "body_type", "transmission",
                 "car_rating", "total_owners"]
fig, axes = plt.subplots(1, len(cat_plot_cols), figsize=(18, 5))
fig.suptitle("Frequency of Key Categorical Variables", fontsize=13, fontweight="bold")

for ax, col in zip(axes, cat_plot_cols):
    order = df[col].value_counts().index
    counts = df[col].value_counts()
    ax.bar(order.astype(str), counts.values, color="#3b82d4", edgecolor="white")
    ax.set_title(col.replace("_", " ").title())
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

plt.tight_layout()
plt.savefig("plot_04_categorical_frequencies.png")
plt.show()
print("Saved: plot_04_categorical_frequencies.png")


# ------------------------------------------------------------------
# 6.5  Bivariate - Sale Price vs Car Age (yr_mfr)
# ------------------------------------------------------------------
median_by_year = df.groupby("yr_mfr")[TARGET].median().reset_index()
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(median_by_year["yr_mfr"], median_by_year[TARGET] / 1e5,
        marker="o", linewidth=2, color="#3b82d4")
ax.set_title("Median Sale Price by Year of Manufacture",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Year of Manufacture")
ax.set_ylabel("Median Sale Price (Rs. Lakhs)")
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("plot_05_price_vs_year.png")
plt.show()
print("Saved: plot_05_price_vs_year.png")


# ------------------------------------------------------------------
# 6.6  Bivariate - Sale Price vs Kilometres Driven
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df["kms_run"] / 1000, df[TARGET] / 1e5,
           alpha=0.15, s=10, color="#3b82d4")
ax.set_title("Sale Price vs Kilometres Driven",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Kilometres Driven (Thousands)")
ax.set_ylabel("Sale Price (Rs. Lakhs)")
plt.tight_layout()
plt.savefig("plot_06_price_vs_kms.png")
plt.show()
print("Saved: plot_06_price_vs_kms.png")


# ------------------------------------------------------------------
# 6.7  Bivariate - Sale Price by Manufacturer (Top 12)
# ------------------------------------------------------------------
top_makes = df["make"].value_counts().head(12).index
df_top = df[df["make"].isin(top_makes)]
median_make = df_top.groupby("make")[TARGET].median().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(13, 6))
ax.bar(median_make.index, median_make.values / 1e5,
       color=sns.color_palette("muted", len(median_make)))
ax.set_title("Median Sale Price by Manufacturer (Top 12 by Listing Volume)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Manufacturer")
ax.set_ylabel("Median Sale Price (Rs. Lakhs)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("plot_07_price_by_make.png")
plt.show()
print("Saved: plot_07_price_by_make.png")


# ------------------------------------------------------------------
# 6.8  Bivariate - Sale Price by Fuel Type
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
fuel_order = df.groupby("fuel_type")[TARGET].median().sort_values(ascending=False).index
sns.boxplot(data=df, x="fuel_type", y=TARGET, order=fuel_order,
            palette="muted", ax=ax, showfliers=False)
ax.set_title("Sale Price Distribution by Fuel Type",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Fuel Type")
ax.set_ylabel("Sale Price (Rs.)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(inr_fmt))
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("plot_08_price_by_fuel.png")
plt.show()
print("Saved: plot_08_price_by_fuel.png")


# ------------------------------------------------------------------
# 6.9  Bivariate - Sale Price by Transmission
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
sns.boxplot(data=df, x="transmission", y=TARGET,
            palette="muted", ax=ax, showfliers=False)
ax.set_title("Sale Price by Transmission Type",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Transmission")
ax.set_ylabel("Sale Price (Rs.)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(inr_fmt))
plt.tight_layout()
plt.savefig("plot_09_price_by_transmission.png")
plt.show()
print("Saved: plot_09_price_by_transmission.png")


# ------------------------------------------------------------------
# 6.10  Bivariate - Sale Price by Number of Previous Owners
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
owner_order = sorted(df["total_owners"].unique())
sns.boxplot(data=df, x="total_owners", y=TARGET, order=owner_order,
            palette="muted", ax=ax, showfliers=False)
ax.set_title("Sale Price by Number of Previous Owners",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Total Previous Owners")
ax.set_ylabel("Sale Price (Rs.)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(inr_fmt))
plt.tight_layout()
plt.savefig("plot_10_price_by_owners.png")
plt.show()
print("Saved: plot_10_price_by_owners.png")


# ------------------------------------------------------------------
# 6.11  Bivariate - Sale Price by City
# ------------------------------------------------------------------
city_order = df.groupby("city")[TARGET].median().sort_values(ascending=False).index
fig, ax = plt.subplots(figsize=(12, 5))
sns.boxplot(data=df, x="city", y=TARGET, order=city_order,
            palette="muted", ax=ax, showfliers=False)
ax.set_title("Sale Price Distribution by City",
             fontsize=13, fontweight="bold")
ax.set_xlabel("City")
ax.set_ylabel("Sale Price (Rs.)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(inr_fmt))
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("plot_11_price_by_city.png")
plt.show()
print("Saved: plot_11_price_by_city.png")


# ------------------------------------------------------------------
# 6.12  Bivariate - Sale Price by Car Rating
# ------------------------------------------------------------------
rating_order = ["great", "good", "fair", "overpriced"]
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x="car_rating", y=TARGET, order=rating_order,
            palette="muted", ax=ax, showfliers=False)
ax.set_title("Sale Price by Car Rating",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Car Rating")
ax.set_ylabel("Sale Price (Rs.)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(inr_fmt))
plt.tight_layout()
plt.savefig("plot_12_price_by_rating.png")
plt.show()
print("Saved: plot_12_price_by_rating.png")


# ------------------------------------------------------------------
# 6.13  Correlation Heatmap
# ------------------------------------------------------------------
num_corr_cols = ["sale_price", "yr_mfr", "kms_run", "total_owners"]
corr_matrix = df[num_corr_cols].corr()

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            square=True, linewidths=0.5, ax=ax)
ax.set_title("Correlation Matrix - Numerical Features",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plot_13_correlation_heatmap.png")
plt.show()
print("Saved: plot_13_correlation_heatmap.png")

# -----------------------------------------------------------------------
# Correlation Interpretation:
# yr_mfr  +ve with sale_price (r approx 0.44): newer cars command higher prices.
# kms_run -ve with sale_price (r approx -0.34): higher mileage lowers price.
# total_owners -ve (r approx -0.22): more owners slightly reduces price.
# Note: correlation indicates association, not causation.
# -----------------------------------------------------------------------


# =============================================================================
# SECTION 7 - BUSINESS / MARKET INSIGHTS
# =============================================================================

print("\n" + "=" * 65)
print("  MARKET INSIGHTS")
print("=" * 65)

# Insight 1: Top manufacturers by listing volume
print("\n1. Manufacturer Share (Top 10 by Listing Count):")
make_counts = df["make"].value_counts().head(10)
for make, cnt in make_counts.items():
    pct = cnt / len(df) * 100
    print(f"   {make:<20} {cnt:>5} listings  ({pct:.1f}%)")

# Insight 2: Highest average price manufacturers (min 20 listings)
print("\n2. Average Sale Price by Manufacturer (min 20 listings):")
avg_price_make = (df.groupby("make")[TARGET].mean()
                  .loc[df.groupby("make")[TARGET].count() >= 20]
                  .sort_values(ascending=False).head(10))
for make, price in avg_price_make.items():
    print(f"   {make:<20} Rs.{price:>10,.0f}")

# Insight 3: Fuel type share
print("\n3. Fuel Type Distribution:")
for ft, cnt in df["fuel_type"].value_counts().items():
    print(f"   {ft:<20} {cnt:>5} ({cnt/len(df)*100:.1f}%)")

# Insight 4: Transmission impact
print("\n4. Average Price by Transmission:")
for tr, grp in df.groupby("transmission"):
    print(f"   {tr:<12} Rs.{grp[TARGET].mean():>10,.0f}  (n={len(grp):,})")

# Insight 5: Price depreciation by age
df["car_age_insight"] = REFERENCE_YEAR - df["yr_mfr"]
price_by_age = df.groupby("car_age_insight")[TARGET].median()
print("\n5. Median Price by Car Age (selected years):")
for age in [1, 3, 5, 7, 10, 15]:
    if age in price_by_age.index:
        print(f"   {age:>2}-year-old car: Rs.{price_by_age[age]:>10,.0f}")

# Insight 6: Owner count impact
print("\n6. Median Price by Number of Previous Owners:")
for owners, grp in df.groupby("total_owners"):
    print(f"   {owners} owner(s): Rs.{grp[TARGET].median():>10,.0f}  (n={len(grp):,})")

# Insight 7: City-wise median price
print("\n7. Median Sale Price by City (sorted):")
city_med = df.groupby("city")[TARGET].median().sort_values(ascending=False)
for city, price in city_med.items():
    print(f"   {city:<15} Rs.{price:>10,.0f}")

# Insight 8: Car rating vs price
print("\n8. Median Price by Car Rating:")
for rating in ["great", "good", "fair", "overpriced"]:
    subset = df[df["car_rating"] == rating]
    if len(subset) > 0:
        print(f"   {rating:<12} Rs.{subset[TARGET].median():>10,.0f}  (n={len(subset):,})")


# =============================================================================
# SECTION 8 - FEATURE ENGINEERING
# =============================================================================

# 8.1  Car Age
# Using REFERENCE_YEAR = 2021 because the vast majority of ads (86%) were
# created in 2021.  This ensures car_age reflects age at time of listing.
df["car_age"] = REFERENCE_YEAR - df["yr_mfr"]

# 8.2  Kilometres Driven Per Year
# Represents average annual usage - a more normalised mileage signal.
# Avoid division by zero for brand-new cars listed in the same year.
df["kms_per_year"] = df["kms_run"] / df["car_age"].replace(0, 1)

print("\n--- Feature Engineering ---")
print(f"car_age        : {df['car_age'].describe()['min']:.0f} - "
      f"{df['car_age'].describe()['max']:.0f} years")
print(f"kms_per_year   : mean = {df['kms_per_year'].mean():,.0f} km/year")

# Add engineered features to the numerical feature list
NUM_FEATURES_ENG = ["car_age", "kms_run", "kms_per_year", "total_owners"]

# Note: yr_mfr is dropped in favour of car_age to avoid redundancy.


# =============================================================================
# SECTION 9 - LEAKAGE ANALYSIS (DOCUMENTED)
# =============================================================================

print("\n" + "=" * 65)
print("  DATA LEAKAGE ANALYSIS")
print("=" * 65)
print("""
EXCLUDED COLUMNS AND REASONING
--------------------------------
broker_quote       : Correlation with sale_price = 0.96.  This is the
                     dealer's internal valuation estimate which is derived
                     from the same pricing process as sale_price.  Using it
                     would cause extreme data leakage - the model would
                     essentially learn to reproduce the broker's formula,
                     not the underlying car attributes.

emi_starts_from    : Correlation approx 1.0.  EMI is literally calculated as
                     (sale_price * factor).  Including this would make
                     prediction trivially easy but completely useless.

booking_down_pymnt : Correlation approx 1.0.  Same reasoning as emi_starts_from.

original_price     : ~44% missing AND represents the showroom price which
                     is often unavailable at prediction time. Also correlated
                     with sale_price by construction.

times_viewed       : Generated AFTER the listing is published; unknown at
                     prediction time for a new listing.

is_hot             : Post-listing engagement indicator; same reasoning.

reserved           : Status assigned after buyer interaction; not available
                     before listing.

car_name           : High-cardinality free-text field that essentially encodes
                     make + model + variant.  Already captured separately.

variant            : Very high cardinality (943 unique); overlaps with make
                     and model; impractical for generalisation.

rto                : RTO code is highly granular (261 unique) and largely
                     redundant with registered_state + city.

registered_city    : 243 unique values - redundant with 'city' (13 values)
                     which covers the selling location more cleanly.

model              : 185 unique values; redundant once 'make' is included.
""")


# =============================================================================
# SECTION 10 - PREPARE DATA FOR MACHINE LEARNING
# =============================================================================

# Final feature sets for the ML model
ML_NUM = ["car_age", "kms_run", "kms_per_year", "total_owners"]
ML_CAT = ["fuel_type", "body_type", "transmission",
          "city", "registered_state", "car_rating",
          "make", "source", "car_availability"]
# Note: boolean columns (assured_buy, fitness_certificate, warranty_avail)
# are treated as categorical so the pipeline handles them uniformly.
ML_BOOL = ["assured_buy", "fitness_certificate", "warranty_avail"]
ML_CAT_ALL = ML_CAT + ML_BOOL

# Combine and prepare X, y
FEATURES = ML_NUM + ML_CAT_ALL
X = df[FEATURES].copy()
y = df[TARGET].copy()

# Convert booleans to strings so OrdinalEncoder handles them consistently
for col in ML_BOOL:
    X[col] = X[col].astype(str)

print(f"\nFeature matrix shape : {X.shape}")
print(f"Target vector shape  : {y.shape}")

# Train / Test split - 80 / 20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE)
print(f"Training set size    : {X_train.shape[0]:,}")
print(f"Test set size        : {X_test.shape[0]:,}")

# Preprocessing Pipelines
# Numerical: impute median (robust to outliers) -> standard scale
num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler())
])

# Categorical: impute with constant 'unknown' -> ordinal encode
cat_pipeline = Pipeline([
    ("impute",  SimpleImputer(strategy="constant", fill_value="unknown")),
    ("encode",  OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, ML_NUM),
    ("cat", cat_pipeline, ML_CAT_ALL)
])

# Note: ColumnTransformer is fit only on X_train inside each Pipeline,
# preventing any leakage of test-set statistics into preprocessing.


# =============================================================================
# SECTION 11 - TRAIN MULTIPLE REGRESSION MODELS
# =============================================================================

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree":     DecisionTreeRegressor(max_depth=10,
                                                random_state=RANDOM_STATE),
    "Random Forest":     RandomForestRegressor(n_estimators=150,
                                               max_depth=15,
                                               n_jobs=-1,
                                               random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200,
                                                    learning_rate=0.1,
                                                    max_depth=5,
                                                    random_state=RANDOM_STATE)
}

results = {}
trained_models = {}

print("\n--- Training Models ---")
for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model",         model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
    trained_models[name] = (pipeline, y_pred)
    print(f"  {name:<22} MAE={mae:>10,.0f}  RMSE={rmse:>10,.0f}  R2={r2:.4f}")


# =============================================================================
# SECTION 12 - MODEL EVALUATION
# =============================================================================

print("\n" + "=" * 65)
print("  MODEL EVALUATION SUMMARY")
print("=" * 65)
print(f"\n{'Model':<24} {'MAE (Rs.)':>12} {'RMSE (Rs.)':>12} {'R2':>8}")
print("-" * 60)
for name, m in results.items():
    print(f"{name:<24} {m['MAE']:>12,.0f} {m['RMSE']:>12,.0f} {m['R2']:>8.4f}")

# -----------------------------------------------------------------------
# Metric Explanations:
# MAE  (Mean Absolute Error)     - average prediction error in rupees.
#                                  Easier to interpret; treats all errors equally.
# RMSE (Root Mean Squared Error) - penalises large errors more heavily.
#                                  Same unit as sale_price (Rs.).
# R2   (Coefficient of Determination) - proportion of variance in sale_price
#                                  explained by the model.  1.0 is perfect;
#                                  0.0 means the model does no better than
#                                  predicting the mean price every time.
# We select the best model using a balanced view of all three metrics.
# -----------------------------------------------------------------------

# Visualise comparison
metrics_df = pd.DataFrame(results).T.reset_index()
metrics_df.rename(columns={"index": "Model"}, inplace=True)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Model Comparison on Test Set", fontsize=14, fontweight="bold")

for ax, metric, color in zip(axes,
                              ["MAE", "RMSE", "R2"],
                              ["#3b82d4", "#7c5cd8", "#22a863"]):
    ax.bar(metrics_df["Model"], metrics_df[metric].astype(float), color=color)
    ax.set_title(metric)
    ax.set_ylabel(metric + " (Rs.)" if metric != "R2" else metric)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    for i, val in enumerate(metrics_df[metric].astype(float)):
        label = f"{val:,.0f}" if metric != "R2" else f"{val:.3f}"
        ax.text(i, val * 1.01, label, ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("plot_14_model_comparison.png")
plt.show()
print("Saved: plot_14_model_comparison.png")


# =============================================================================
# SECTION 13 - ACTUAL VS PREDICTED & RESIDUAL ANALYSIS
# =============================================================================

# Identify best model by R2
best_model_name = max(results, key=lambda k: results[k]["R2"])
best_pipeline, best_pred = trained_models[best_model_name]
print(f"\nBest Model: {best_model_name}  (R2 = {results[best_model_name]['R2']:.4f})")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"Best Model: {best_model_name} - Test Set",
             fontsize=14, fontweight="bold")

# Actual vs Predicted
axes[0].scatter(y_test / 1e5, best_pred / 1e5, alpha=0.25, s=12,
                color="#3b82d4")
lim = max(y_test.max(), best_pred.max()) / 1e5
axes[0].plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Perfect fit")
axes[0].set_xlabel("Actual Price (Rs. Lakhs)")
axes[0].set_ylabel("Predicted Price (Rs. Lakhs)")
axes[0].set_title("Actual vs Predicted Sale Price")
axes[0].legend()

# Residuals
residuals = y_test.values - best_pred
axes[1].scatter(best_pred / 1e5, residuals / 1e5, alpha=0.25, s=12,
                color="#7c5cd8")
axes[1].axhline(0, color="red", linestyle="--", linewidth=1.5)
axes[1].set_xlabel("Predicted Price (Rs. Lakhs)")
axes[1].set_ylabel("Residual (Actual - Predicted, Rs. Lakhs)")
axes[1].set_title("Residual Plot")

plt.tight_layout()
plt.savefig("plot_15_actual_vs_predicted.png")
plt.show()
print("Saved: plot_15_actual_vs_predicted.png")

# -----------------------------------------------------------------------
# Interpretation:
# - Points clustering tightly around the red diagonal line indicate good
#   predictions.
# - The residual plot should show a random scatter around zero; systematic
#   patterns would indicate model bias or missing features.
# -----------------------------------------------------------------------


# =============================================================================
# SECTION 14 - FEATURE IMPORTANCE / MODEL INTERPRETATION
# =============================================================================

# Feature importance is available for tree-based models.
# We use the best tree-based model.
tree_models = ["Random Forest", "Gradient Boosting", "Decision Tree"]
importance_model_name = next((m for m in [best_model_name] + tree_models
                               if m in trained_models), None)

importance_pipeline = trained_models[importance_model_name][0]
imp_model = importance_pipeline.named_steps["model"]

# Reconstruct feature names after ColumnTransformer
feature_names_out = (ML_NUM +
                     list(importance_pipeline.named_steps["preprocessor"]
                          .named_transformers_["cat"]
                          .named_steps["encode"]
                          .get_feature_names_out(ML_CAT_ALL)))

importances = imp_model.feature_importances_
feat_imp = (pd.Series(importances, index=feature_names_out)
              .sort_values(ascending=False)
              .head(20))

fig, ax = plt.subplots(figsize=(11, 7))
ax.barh(feat_imp.index[::-1], feat_imp.values[::-1], color="#3b82d4")
ax.set_title(f"Top 20 Feature Importances - {importance_model_name}",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Importance (Gini / Impurity Reduction)")
plt.tight_layout()
plt.savefig("plot_16_feature_importance.png")
plt.show()
print(f"Saved: plot_16_feature_importance.png")

print(f"\nTop 10 features ({importance_model_name}):")
for feat, imp in feat_imp.head(10).items():
    print(f"  {feat:<35} {imp:.4f}")

# -----------------------------------------------------------------------
# Note: Feature importance indicates how much each feature contributes
# to the model's predictions.  A high importance score means the feature
# is strongly associated with sale_price within this model.  This does NOT
# imply that the feature causes price changes - correlation != causation.
# -----------------------------------------------------------------------


# =============================================================================
# SECTION 15 - EXAMPLE PRICE PREDICTION
# =============================================================================

print("\n" + "=" * 65)
print("  EXAMPLE PRICE PREDICTION (from test set)")
print("=" * 65)

# Pick the sample with smallest prediction error from the test set
abs_errors = np.abs(y_test.values - best_pred)
best_idx_in_test = np.argmin(abs_errors)

sample_X = X_test.iloc[[best_idx_in_test]]
actual_price = y_test.iloc[best_idx_in_test]
predicted_price = best_pipeline.predict(sample_X)[0]
error = actual_price - predicted_price

# Retrieve original row for display
original_row = df.loc[sample_X.index[0]]

print("\nCar Details:")
display_fields = ["car_name", "yr_mfr", "fuel_type", "transmission",
                  "body_type", "kms_run", "total_owners", "city", "car_rating",
                  "car_age", "kms_per_year"]
for field in display_fields:
    if field in original_row.index:
        print(f"  {field:<20}: {original_row[field]}")

print(f"\n  {'Actual Sale Price':<20}: Rs.{actual_price:>12,.0f}")
print(f"  {'Predicted Price':<20}: Rs.{predicted_price:>12,.0f}")
print(f"  {'Prediction Error':<20}: Rs.{error:>12,.0f}  "
      f"({abs(error)/actual_price*100:.1f}% off)")


# =============================================================================
# SECTION 16 - FINAL PROJECT SUMMARY
# =============================================================================

print("\n" + "=" * 65)
print("  FINAL PROJECT SUMMARY")
print("=" * 65)

print(f"""
DATASET SUMMARY
---------------
Records          : {len(df):,} used-car listings
Features         : {len(FEATURES)} predictive features (after leakage removal)
Target Variable  : sale_price (listing price in INR)
Data Source      : Indian online used-car marketplace
Ad Period        : 2019-2021 (86% from 2021)
Cities Covered   : {df['city'].nunique()} major Indian cities
Manufacturers    : {df['make'].nunique()} brands

TOP MARKET INSIGHTS
-------------------
1. Maruti and Hyundai dominate listings with ~43% and ~24% market share
   respectively, reflecting their strong presence in the Indian market.

2. Newer cars command significantly higher prices - median price for a
   1-year-old car is roughly 2-3x higher than a 10-year-old car.

3. Automatic cars are priced substantially higher than manual cars,
   reflecting the premium associated with convenience.

4. Diesel cars show higher median prices than petrol cars, likely due to
   their use in higher-end / larger vehicles.

5. Cars with 1 previous owner are priced higher than cars with 2+ owners;
   price drops noticeably with each additional ownership change.

6. Mumbai, Bangalore, and New Delhi show the highest average used-car
   prices among the 13 cities in the dataset.

7. Car rating tracks price: 'great' rated cars have the highest prices;
   'overpriced' rated cars have the highest absolute prices but poor
   value perception.

8. Higher mileage (kms_run) is negatively associated with price, as
   expected - heavier usage reduces resale value.
""")

print("MODEL COMPARISON")
print("-" * 60)
print(f"{'Model':<24} {'MAE (Rs.)':>12} {'RMSE (Rs.)':>12} {'R2':>8}")
print("-" * 60)
for name, m in results.items():
    marker = " < BEST" if name == best_model_name else ""
    print(f"{name:<24} {m['MAE']:>12,.0f} {m['RMSE']:>12,.0f} "
          f"{m['R2']:>8.4f}{marker}")

print(f"""
BEST PERFORMING MODEL
---------------------
Model : {best_model_name}
MAE   : Rs.{results[best_model_name]['MAE']:,.0f}
RMSE  : Rs.{results[best_model_name]['RMSE']:,.0f}
R2    : {results[best_model_name]['R2']:.4f}

LIMITATIONS
-----------
1. Dataset size (7,400 listings) is modest; a larger dataset would improve
   model generalisation.
2. Data covers only one platform/source and may not represent the entire
   Indian used-car market.
3. Listings may include asking prices, not final transaction prices.
4. Luxury brands (Mercedes, BMW, Audi) are under-represented (<1% each),
   so predictions for premium cars may be less accurate.
5. The dataset spans 2019-2021; price dynamics have shifted due to supply
   disruptions and changing consumer preferences post-2021.
6. Geographic coverage is limited to 13 cities; rural and tier-2/3 markets
   are not represented.

FUTURE IMPROVEMENTS
-------------------
1. Incorporate larger, more recent datasets (2022-2024).
2. Add more vehicle specifications: engine displacement, mileage (km/L),
   safety ratings, service history.
3. Perform hyperparameter tuning (GridSearchCV or RandomizedSearchCV).
4. Experiment with advanced models: XGBoost, LightGBM, CatBoost.
5. Include macroeconomic indicators: fuel prices, loan interest rates.
6. Build a web application (Flask / Streamlit) for real-time price queries.
7. Apply geospatial features for more granular location-based pricing.
""")

print("=" * 65)
print("  PROJECT COMPLETE")
print("=" * 65)
