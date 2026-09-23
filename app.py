# =============================================================================
# Streamlit Frontend
# Used Car Market Analysis and Price Prediction in India
# IBM x Bharat Cares Data Analysis Internship Project
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Used Car Market – India",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

RANDOM_STATE = 42
REFERENCE_YEAR = 2021

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)
plt.rcParams.update({"figure.dpi": 110, "figure.facecolor": "white"})

# ── Helper ────────────────────────────────────────────────────────────────────
def inr_fmt(x, _):
    if x >= 1e5:
        return f"Rs.{x/1e5:.1f}L"
    if x >= 1e3:
        return f"Rs.{x/1e3:.0f}K"
    return f"Rs.{x:.0f}"

def lakh(x):
    return f"Rs. {x/1e5:.2f} L"

# ── Data loading & caching ────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading and cleaning dataset…")
def load_data():
    df = pd.read_csv("Used_Car_Price_Prediction.csv")

    # --- cleaning ---
    df.drop_duplicates(inplace=True)
    df = df[df["sale_price"] > 0].copy()
    df["ad_created_on"] = pd.to_datetime(df["ad_created_on"], errors="coerce")

    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    str_cols = [c for c in (df.select_dtypes(include="object").columns.tolist() +
                             df.select_dtypes(include="string").columns.tolist())
                if c not in bool_cols]
    for col in str_cols:
        df[col] = df[col].astype(str).str.lower().str.strip().replace("nan", np.nan)

    for col in ["body_type", "transmission", "source", "registered_city",
                "registered_state", "car_availability", "car_rating"]:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    df["fitness_certificate"] = df["fitness_certificate"].fillna(False)
    df["kms_run"] = df["kms_run"].clip(upper=df["kms_run"].quantile(0.99))

    # --- feature engineering ---
    df["car_age"] = REFERENCE_YEAR - df["yr_mfr"]
    df["kms_per_year"] = df["kms_run"] / df["car_age"].replace(0, 1)

    return df


@st.cache_resource(show_spinner="Training models… (runs once)")
def train_models(df):
    ML_NUM = ["car_age", "kms_run", "kms_per_year", "total_owners"]
    ML_CAT = ["fuel_type", "body_type", "transmission", "city",
              "registered_state", "car_rating", "make", "source", "car_availability"]
    ML_BOOL = ["assured_buy", "fitness_certificate", "warranty_avail"]
    ML_CAT_ALL = ML_CAT + ML_BOOL
    FEATURES = ML_NUM + ML_CAT_ALL

    X = df[FEATURES].copy()
    y = df["sale_price"].copy()
    for col in ML_BOOL:
        X[col] = X[col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE)

    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("scl", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="unknown")),
                          ("enc", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                 unknown_value=-1))])
    pre = ColumnTransformer([("num", num_pipe, ML_NUM),
                              ("cat", cat_pipe, ML_CAT_ALL)])

    model_defs = {
        "Linear Regression":  LinearRegression(),
        "Decision Tree":      DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE),
        "Random Forest":      RandomForestRegressor(n_estimators=150, max_depth=15,
                                                    n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting":  GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
                                                         max_depth=5, random_state=RANDOM_STATE),
    }

    results, pipelines, preds = {}, {}, {}
    for name, mdl in model_defs.items():
        pipe = Pipeline([("pre", pre), ("mdl", mdl)])
        pipe.fit(X_train, y_train)
        yp = pipe.predict(X_test)
        results[name] = {
            "MAE":  mean_absolute_error(y_test, yp),
            "RMSE": np.sqrt(mean_squared_error(y_test, yp)),
            "R2":   r2_score(y_test, yp),
        }
        pipelines[name] = pipe
        preds[name] = yp

    best = max(results, key=lambda k: results[k]["R2"])

    # feature names for importance
    best_pipe = pipelines[best]
    feat_names = ML_NUM + list(
        best_pipe.named_steps["pre"].named_transformers_["cat"]
                 .named_steps["enc"].get_feature_names_out(ML_CAT_ALL))
    imp_vals = best_pipe.named_steps["mdl"].feature_importances_
    feat_imp = pd.Series(imp_vals, index=feat_names).sort_values(ascending=False)

    return {
        "results":   results,
        "pipelines": pipelines,
        "preds":     preds,
        "best":      best,
        "X_test":    X_test,
        "y_test":    y_test,
        "feat_imp":  feat_imp,
        "FEATURES":  FEATURES,
        "ML_NUM":    ML_NUM,
        "ML_CAT_ALL": ML_CAT_ALL,
    }


# ── Load data ─────────────────────────────────────────────────────────────────
df = load_data()
tm = train_models(df)

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg",
    width=90,
)
st.sidebar.markdown("## 🚗 Used Car India")
st.sidebar.markdown("IBM × Bharat Cares Internship")
st.sidebar.markdown("---")

pages = [
    "🏠 Overview",
    "🔍 Dataset Explorer",
    "📊 EDA & Market Insights",
    "🤖 Model Evaluation",
    "💰 Price Predictor",
]
page = st.sidebar.radio("Navigate to", pages, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.caption(
    f"**Dataset:** {len(df):,} listings · 29 columns\n\n"
    f"**Target:** sale\\_price (INR)\n\n"
    f"**Best model:** {tm['best']}\n\n"
    f"**R²:** {tm['results'][tm['best']]['R2']:.4f}"
)

# =============================================================================
# PAGE 1 – OVERVIEW
# =============================================================================
if page == "🏠 Overview":
    st.title("Used Car Market Analysis & Price Prediction in India")
    st.caption("IBM × Bharat Cares Data Analysis Internship Project")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Listings", f"{len(df):,}")
    c2.metric("Manufacturers", f"{df['make'].nunique()}")
    c3.metric("Cities Covered", f"{df['city'].nunique()}")
    c4.metric("Best Model R²", f"{tm['results'][tm['best']]['R2']:.4f}")

    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Project Goal")
        st.markdown("""
This project performs a **complete data science workflow** on a real Indian used-car listing dataset:

- **Market Analysis** – Understand pricing patterns by manufacturer, fuel type, transmission, city, and ownership history.
- **Price Prediction** – Train and compare four regression models to predict a car's selling price from its attributes.

The project was built as part of the **IBM × Bharat Cares Data Analysis Internship**.
        """)

        st.subheader("Dataset Overview")
        st.markdown("""
| Property | Value |
|---|---|
| Source | Indian used-car marketplace |
| Period | 2019 – 2021 (86% from 2021) |
| Records | 7,396 (after cleaning) |
| Raw columns | 29 |
| ML features | 16 (leakage-free) |
| Target | `sale_price` (INR) |
        """)

    with col_r:
        st.subheader("Key Market Insights")
        insights = [
            "Maruti (43%) and Hyundai (24%) dominate listings.",
            "Automatic cars command a ~74% price premium over manual.",
            "Price halves every ~7–8 years of car age.",
            "First-owner cars earn ~17% more than second-owner.",
            "Diesel cars have higher median prices than petrol.",
            "Chennai records the highest median used-car price.",
            "'Great'-rated cars cost ~60% more than 'good'-rated ones.",
            "Higher mileage (kms_run) negatively correlates with price.",
        ]
        for i, ins in enumerate(insights, 1):
            st.markdown(f"**{i}.** {ins}")

        st.subheader("Model Performance")
        res_df = pd.DataFrame(tm["results"]).T.reset_index()
        res_df.columns = ["Model", "MAE (Rs.)", "RMSE (Rs.)", "R²"]
        res_df["MAE (Rs.)"]  = res_df["MAE (Rs.)"].map(lambda x: f"{x:,.0f}")
        res_df["RMSE (Rs.)"] = res_df["RMSE (Rs.)"].map(lambda x: f"{x:,.0f}")
        res_df["R²"]         = res_df["R²"].map(lambda x: f"{x:.4f}")
        st.dataframe(res_df, use_container_width=True, hide_index=True)

# =============================================================================
# PAGE 2 – DATASET EXPLORER
# =============================================================================
elif page == "🔍 Dataset Explorer":
    st.title("Dataset Explorer")
    st.markdown("Inspect the raw and cleaned dataset interactively.")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📋 Browse Data", "📈 Statistics", "🧹 Missing Values"])

    with tab1:
        st.subheader("Dataset Preview")

        col_f, col_r2 = st.columns([2, 1])
        with col_f:
            makes = ["All"] + sorted(df["make"].dropna().unique().tolist())
            sel_make = st.selectbox("Filter by Manufacturer", makes)
        with col_r2:
            n_rows = st.slider("Rows to show", 5, 100, 20)

        disp = df if sel_make == "All" else df[df["make"] == sel_make]
        st.dataframe(disp.head(n_rows), use_container_width=True)
        st.caption(f"Showing {min(n_rows, len(disp))} of {len(disp):,} rows")

    with tab2:
        st.subheader("Numerical Statistics")
        num_cols = ["sale_price", "yr_mfr", "kms_run", "total_owners",
                    "car_age", "kms_per_year"]
        st.dataframe(df[num_cols].describe().round(2), use_container_width=True)

        st.subheader("Unique Values per Column")
        uniq = pd.DataFrame({
            "Column": df.columns,
            "Unique Values": [df[c].nunique() for c in df.columns],
            "Missing": [df[c].isnull().sum() for c in df.columns],
            "Missing %": [(df[c].isnull().sum() / len(df) * 100).round(2) for c in df.columns],
        })
        st.dataframe(uniq, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Missing Value Summary")
        mv = df.isnull().sum()
        mv_pct = (mv / len(df) * 100).round(2)
        mv_df = pd.DataFrame({"Missing Count": mv, "Missing %": mv_pct})
        mv_df = mv_df[mv_df["Missing Count"] > 0].sort_values("Missing Count", ascending=False)
        if len(mv_df) == 0:
            st.success("No missing values in the cleaned dataset.")
        else:
            st.dataframe(mv_df, use_container_width=True)
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.barh(mv_df.index, mv_df["Missing %"], color="#3b82d4")
            ax.set_xlabel("Missing %")
            ax.set_title("Missing Values by Column")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# =============================================================================
# PAGE 3 – EDA & MARKET INSIGHTS
# =============================================================================
elif page == "📊 EDA & Market Insights":
    st.title("Exploratory Data Analysis & Market Insights")
    st.markdown("---")

    tab_uni, tab_bi, tab_corr, tab_insights = st.tabs([
        "📦 Univariate", "📉 Bivariate", "🔥 Correlation", "💡 Insights"
    ])

    # --- Univariate ---
    with tab_uni:
        st.subheader("Sale Price Distribution")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(df["sale_price"] / 1e5, bins=60, color="#3b82d4", edgecolor="white")
        axes[0].set_xlabel("Sale Price (Rs. Lakhs)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Histogram of Sale Price")
        axes[1].hist(np.log1p(df["sale_price"]), bins=60, color="#7c5cd8", edgecolor="white")
        axes[1].set_xlabel("log(Sale Price)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Log-Transformed (more symmetric)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader("Year of Manufacture Distribution")
        year_counts = df["yr_mfr"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.bar(year_counts.index.astype(str), year_counts.values,
               color="#3b82d4", edgecolor="white")
        ax.set_xlabel("Year of Manufacture")
        ax.set_ylabel("Listings")
        ax.set_title("Listings by Year of Manufacture")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader("Kilometres Driven Distribution")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(df["kms_run"] / 1000, bins=50, color="#3b82d4", edgecolor="white")
        ax.set_xlabel("Kilometres Driven (Thousands)")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Kilometres Driven")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader("Categorical Variable Frequencies")
        cat_cols = ["fuel_type", "body_type", "transmission", "car_rating"]
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, col in zip(axes, cat_cols):
            vc = df[col].value_counts()
            ax.bar(vc.index.astype(str), vc.values, color="#3b82d4", edgecolor="white")
            ax.set_title(col.replace("_", " ").title())
            ax.set_ylabel("Count")
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # --- Bivariate ---
    with tab_bi:
        chart_choice = st.selectbox("Select chart", [
            "Price vs Year of Manufacture",
            "Price vs Kilometres Driven",
            "Price by Manufacturer (Top 12)",
            "Price by Fuel Type",
            "Price by Transmission",
            "Price by Number of Owners",
            "Price by City",
            "Price by Car Rating",
        ])

        if chart_choice == "Price vs Year of Manufacture":
            med = df.groupby("yr_mfr")["sale_price"].median().reset_index()
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.plot(med["yr_mfr"], med["sale_price"] / 1e5,
                    marker="o", linewidth=2, color="#3b82d4")
            ax.set_xlabel("Year of Manufacture")
            ax.set_ylabel("Median Sale Price (Rs. Lakhs)")
            ax.set_title("Median Sale Price by Year of Manufacture")
            ax.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        elif chart_choice == "Price vs Kilometres Driven":
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(df["kms_run"] / 1000, df["sale_price"] / 1e5,
                       alpha=0.12, s=8, color="#3b82d4")
            ax.set_xlabel("Kilometres Driven (Thousands)")
            ax.set_ylabel("Sale Price (Rs. Lakhs)")
            ax.set_title("Sale Price vs Kilometres Driven")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        elif chart_choice == "Price by Manufacturer (Top 12)":
            top12 = df["make"].value_counts().head(12).index
            med = (df[df["make"].isin(top12)]
                   .groupby("make")["sale_price"].median()
                   .sort_values(ascending=False))
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.bar(med.index, med.values / 1e5,
                   color=sns.color_palette("muted", len(med)))
            ax.set_xlabel("Manufacturer")
            ax.set_ylabel("Median Sale Price (Rs. Lakhs)")
            ax.set_title("Median Sale Price – Top 12 Manufacturers")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        elif chart_choice == "Price by Fuel Type":
            order = df.groupby("fuel_type")["sale_price"].median().sort_values(ascending=False).index
            fig, ax = plt.subplots(figsize=(9, 4))
            sns.boxplot(data=df, x="fuel_type", y="sale_price", order=order,
                        palette="muted", ax=ax, showfliers=False)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(inr_fmt))
            ax.set_xlabel("Fuel Type")
            ax.set_ylabel("Sale Price")
            ax.set_title("Sale Price Distribution by Fuel Type")
            plt.xticks(rotation=20)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        elif chart_choice == "Price by Transmission":
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.boxplot(data=df, x="transmission", y="sale_price",
                        palette="muted", ax=ax, showfliers=False)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(inr_fmt))
            ax.set_xlabel("Transmission")
            ax.set_ylabel("Sale Price")
            ax.set_title("Sale Price by Transmission Type")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        elif chart_choice == "Price by Number of Owners":
            order = sorted(df["total_owners"].unique())
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(data=df, x="total_owners", y="sale_price", order=order,
                        palette="muted", ax=ax, showfliers=False)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(inr_fmt))
            ax.set_xlabel("Total Previous Owners")
            ax.set_ylabel("Sale Price")
            ax.set_title("Sale Price by Number of Previous Owners")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        elif chart_choice == "Price by City":
            order = df.groupby("city")["sale_price"].median().sort_values(ascending=False).index
            fig, ax = plt.subplots(figsize=(12, 4))
            sns.boxplot(data=df, x="city", y="sale_price", order=order,
                        palette="muted", ax=ax, showfliers=False)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(inr_fmt))
            ax.set_xlabel("City")
            ax.set_ylabel("Sale Price")
            ax.set_title("Sale Price Distribution by City")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        elif chart_choice == "Price by Car Rating":
            order = ["great", "good", "fair", "overpriced"]
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(data=df, x="car_rating", y="sale_price", order=order,
                        palette="muted", ax=ax, showfliers=False)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(inr_fmt))
            ax.set_xlabel("Car Rating")
            ax.set_ylabel("Sale Price")
            ax.set_title("Sale Price by Car Rating")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # --- Correlation ---
    with tab_corr:
        st.subheader("Correlation Matrix – Numerical Features")
        num_cols = ["sale_price", "yr_mfr", "kms_run", "total_owners",
                    "car_age", "kms_per_year"]
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    square=True, linewidths=0.5, ax=ax)
        ax.set_title("Correlation Matrix")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("""
**Interpretation:**
- `yr_mfr` has a **positive** correlation (~0.44) with `sale_price` — newer cars are priced higher.
- `kms_run` has a **negative** correlation (~−0.34) — heavier usage lowers resale value.
- `total_owners` has a **negative** correlation (~−0.22) — more previous owners slightly reduces price.
- `car_age` mirrors `yr_mfr` (negative correlation) as expected.
- Correlation indicates **association**, not causation.
        """)

    # --- Insights ---
    with tab_insights:
        st.subheader("Market Insights from the Data")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Manufacturer Market Share (Top 10)")
            mc = df["make"].value_counts().head(10).reset_index()
            mc.columns = ["Manufacturer", "Listings"]
            mc["Share %"] = (mc["Listings"] / len(df) * 100).round(1)
            st.dataframe(mc, use_container_width=True, hide_index=True)

            st.markdown("#### Average Price by Manufacturer (min 20 listings)")
            avg_p = (df.groupby("make")["sale_price"].agg(["mean", "count"])
                       .query("count >= 20").sort_values("mean", ascending=False)
                       .head(10).reset_index())
            avg_p.columns = ["Manufacturer", "Avg Price (Rs.)", "Listings"]
            avg_p["Avg Price (Rs.)"] = avg_p["Avg Price (Rs.)"].map(lambda x: f"{x:,.0f}")
            st.dataframe(avg_p, use_container_width=True, hide_index=True)

        with col2:
            st.markdown("#### Median Price by Fuel Type")
            ft = (df.groupby("fuel_type")["sale_price"].median()
                    .sort_values(ascending=False).reset_index())
            ft.columns = ["Fuel Type", "Median Price (Rs.)"]
            ft["Median Price (Rs.)"] = ft["Median Price (Rs.)"].map(lambda x: f"{x:,.0f}")
            st.dataframe(ft, use_container_width=True, hide_index=True)

            st.markdown("#### Median Price by Transmission")
            tr = (df.groupby("transmission")["sale_price"].median()
                    .sort_values(ascending=False).reset_index())
            tr.columns = ["Transmission", "Median Price (Rs.)"]
            tr["Median Price (Rs.)"] = tr["Median Price (Rs.)"].map(lambda x: f"{x:,.0f}")
            st.dataframe(tr, use_container_width=True, hide_index=True)

            st.markdown("#### Median Price by City")
            city = (df.groupby("city")["sale_price"].median()
                      .sort_values(ascending=False).reset_index())
            city.columns = ["City", "Median Price (Rs.)"]
            city["Median Price (Rs.)"] = city["Median Price (Rs.)"].map(lambda x: f"{x:,.0f}")
            st.dataframe(city, use_container_width=True, hide_index=True)

# =============================================================================
# PAGE 4 – MODEL EVALUATION
# =============================================================================
elif page == "🤖 Model Evaluation":
    st.title("Model Training & Evaluation")
    st.markdown("---")

    tab_metrics, tab_plots, tab_imp = st.tabs([
        "📊 Metrics", "🎯 Actual vs Predicted", "🏆 Feature Importance"
    ])

    with tab_metrics:
        st.subheader("Model Comparison on Test Set (20% hold-out)")
        st.markdown("""
| Metric | Meaning |
|---|---|
| **MAE** | Mean Absolute Error – average prediction error in rupees (lower is better) |
| **RMSE** | Root Mean Squared Error – penalises large errors more; same unit as price (lower is better) |
| **R²** | Proportion of price variance explained by the model (closer to 1.0 is better) |
        """)

        res_df = pd.DataFrame(tm["results"]).T.reset_index()
        res_df.columns = ["Model", "MAE (Rs.)", "RMSE (Rs.)", "R²"]

        # Highlight best
        def highlight_best(row):
            best_r2 = max(tm["results"][m]["R2"] for m in tm["results"])
            if abs(row["R²"] - best_r2) < 1e-6:
                return ["background-color: #d1fadf"] * len(row)
            return [""] * len(row)

        res_styled = res_df.style.format({
            "MAE (Rs.)":  "{:,.0f}",
            "RMSE (Rs.)": "{:,.0f}",
            "R²":         "{:.4f}",
        }).apply(highlight_best, axis=1)
        st.dataframe(res_styled, use_container_width=True, hide_index=True)

        st.success(f"**Best model: {tm['best']}** — R² = {tm['results'][tm['best']]['R2']:.4f}, "
                   f"MAE = Rs. {tm['results'][tm['best']]['MAE']:,.0f}")

        # Bar chart comparison
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle("Model Comparison", fontweight="bold")
        for ax, metric, color in zip(axes,
                                      ["MAE", "RMSE", "R2"],
                                      ["#3b82d4", "#7c5cd8", "#22a863"]):
            vals = [tm["results"][m][metric] for m in tm["results"]]
            names = list(tm["results"].keys())
            ax.bar(names, vals, color=color)
            ax.set_title(metric)
            ax.set_ylabel(metric + " (Rs.)" if metric != "R2" else metric)
            plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
            for i, v in enumerate(vals):
                lbl = f"{v:,.0f}" if metric != "R2" else f"{v:.3f}"
                ax.text(i, v * 1.01, lbl, ha="center", va="bottom", fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab_plots:
        best = tm["best"]
        y_test = tm["y_test"]
        best_pred = tm["preds"][best]

        st.subheader(f"Actual vs Predicted – {best}")
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        axes[0].scatter(y_test / 1e5, best_pred / 1e5, alpha=0.2, s=10, color="#3b82d4")
        lim = max(y_test.max(), best_pred.max()) / 1e5
        axes[0].plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Perfect fit")
        axes[0].set_xlabel("Actual Price (Rs. Lakhs)")
        axes[0].set_ylabel("Predicted Price (Rs. Lakhs)")
        axes[0].set_title("Actual vs Predicted")
        axes[0].legend()

        residuals = y_test.values - best_pred
        axes[1].scatter(best_pred / 1e5, residuals / 1e5, alpha=0.2, s=10, color="#7c5cd8")
        axes[1].axhline(0, color="red", linestyle="--", linewidth=1.5)
        axes[1].set_xlabel("Predicted Price (Rs. Lakhs)")
        axes[1].set_ylabel("Residual (Rs. Lakhs)")
        axes[1].set_title("Residual Plot")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("""
**Reading these charts:**
- **Actual vs Predicted**: points clustered tightly along the red diagonal = accurate model.
- **Residual Plot**: residuals should scatter randomly around zero; any pattern suggests bias or missing features.
        """)

    with tab_imp:
        st.subheader(f"Feature Importance – {tm['best']}")
        fi = tm["feat_imp"].head(20)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(fi.index[::-1], fi.values[::-1], color="#3b82d4")
        ax.set_xlabel("Importance Score")
        ax.set_title(f"Top 20 Feature Importances ({tm['best']})")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("""
> **Note:** Feature importance shows how much each feature contributes to the model's predictions.
> A high score means the feature is *associated* with price within this model.
> This is **not** the same as causal influence.
        """)

        top10 = ti = tm["feat_imp"].head(10).reset_index()
        top10.columns = ["Feature", "Importance"]
        top10["Importance"] = top10["Importance"].map(lambda x: f"{x:.4f}")
        st.dataframe(top10, use_container_width=True, hide_index=True)

# =============================================================================
# PAGE 5 – PRICE PREDICTOR
# =============================================================================
elif page == "💰 Price Predictor":
    st.title("Used Car Price Predictor")
    st.markdown("Enter the details of a used car to get an estimated selling price.")
    st.markdown("---")

    best_pipe = tm["pipelines"][tm["best"]]

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.subheader("Car Details")

        make_options = sorted(df["make"].dropna().unique().tolist())
        sel_make = st.selectbox("Manufacturer", make_options, index=make_options.index("maruti") if "maruti" in make_options else 0)

        yr = st.slider("Year of Manufacture", int(df["yr_mfr"].min()), int(df["yr_mfr"].max()), 2016)
        kms = st.number_input("Kilometres Driven", min_value=500, max_value=400000, value=45000, step=1000)
        owners = st.selectbox("Number of Previous Owners", sorted(df["total_owners"].unique()), index=0)

        fuel_opts = sorted(df["fuel_type"].dropna().unique().tolist())
        fuel = st.selectbox("Fuel Type", fuel_opts, index=fuel_opts.index("petrol") if "petrol" in fuel_opts else 0)

        body_opts = sorted(df["body_type"].dropna().unique().tolist())
        body = st.selectbox("Body Type", body_opts, index=body_opts.index("hatchback") if "hatchback" in body_opts else 0)

        trans_opts = sorted(df["transmission"].dropna().unique().tolist())
        trans = st.selectbox("Transmission", trans_opts, index=trans_opts.index("manual") if "manual" in trans_opts else 0)

    with col_r:
        st.subheader("Location & Condition")

        city_opts = sorted(df["city"].dropna().unique().tolist())
        city_sel = st.selectbox("City (Listing Location)", city_opts, index=city_opts.index("mumbai") if "mumbai" in city_opts else 0)

        state_opts = sorted(df["registered_state"].dropna().unique().tolist())
        state_sel = st.selectbox("Registered State", state_opts, index=0)

        rating_opts = ["great", "good", "fair", "overpriced"]
        rating_sel = st.selectbox("Car Rating", rating_opts, index=0)

        source_opts = sorted(df["source"].dropna().unique().tolist())
        source_sel = st.selectbox("Sale Source", source_opts, index=0)

        avail_opts = sorted(df["car_availability"].dropna().unique().tolist())
        avail_sel = st.selectbox("Availability", avail_opts, index=avail_opts.index("in_stock") if "in_stock" in avail_opts else 0)

        assured = st.checkbox("Assured Buy", value=True)
        fitness = st.checkbox("Has Fitness Certificate", value=True)
        warranty = st.checkbox("Warranty Available", value=False)

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Price", type="primary", use_container_width=True)

    if predict_btn:
        car_age = REFERENCE_YEAR - yr
        kms_per_year = kms / max(car_age, 1)

        input_data = pd.DataFrame([{
            "car_age":          car_age,
            "kms_run":          float(kms),
            "kms_per_year":     kms_per_year,
            "total_owners":     int(owners),
            "fuel_type":        fuel,
            "body_type":        body,
            "transmission":     trans,
            "city":             city_sel,
            "registered_state": state_sel,
            "car_rating":       rating_sel,
            "make":             sel_make,
            "source":           source_sel,
            "car_availability": avail_sel,
            "assured_buy":      str(assured),
            "fitness_certificate": str(fitness),
            "warranty_avail":   str(warranty),
        }])

        predicted = best_pipe.predict(input_data)[0]
        predicted = max(predicted, 0)

        # Find similar cars in the dataset for context
        similar = df[
            (df["make"] == sel_make) &
            (df["fuel_type"] == fuel) &
            (df["transmission"] == trans) &
            (df["yr_mfr"].between(yr - 2, yr + 2))
        ]["sale_price"]

        col_p1, col_p2, col_p3 = st.columns(3)
        col_p1.metric("🤖 Predicted Price", f"Rs. {predicted/1e5:.2f} Lakhs",
                      f"Rs. {predicted:,.0f}")
        if len(similar) > 0:
            col_p2.metric("📊 Median of Similar Cars",
                          f"Rs. {similar.median()/1e5:.2f} Lakhs",
                          f"{len(similar)} similar listings found")
            col_p3.metric("Range of Similar Cars",
                          f"Rs. {similar.min()/1e5:.1f}L – Rs. {similar.max()/1e5:.1f}L",
                          f"±{similar.std()/1e5:.2f}L std dev")
        else:
            col_p2.info("No similar listings found for comparison.")

        with st.expander("View input summary"):
            st.json({
                "Manufacturer":     sel_make,
                "Year":             yr,
                "Car Age":          f"{car_age} years",
                "Kms Driven":       f"{kms:,} km",
                "Kms/Year":         f"{kms_per_year:,.0f} km/year",
                "Owners":           int(owners),
                "Fuel":             fuel,
                "Body":             body,
                "Transmission":     trans,
                "City":             city_sel,
                "State":            state_sel,
                "Rating":           rating_sel,
                "Model Used":       tm["best"],
                "R² of Model":      f"{tm['results'][tm['best']]['R2']:.4f}",
            })

        st.info(
            f"**Disclaimer:** This is an estimate from a **{tm['best']}** model "
            f"(R² = {tm['results'][tm['best']]['R2']:.4f}, "
            f"average error ≈ Rs. {tm['results'][tm['best']]['MAE']:,.0f}). "
            "Actual market prices depend on vehicle condition, local demand, negotiation, and factors not captured in this dataset."
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("IBM × Bharat Cares Data Analysis Internship | Used Car Market Analysis & Price Prediction in India")
