import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.stattools import omni_normtest, jarque_bera

# ---------- Page config ----------
st.set_page_config(
    page_title="Insurance Charges Dashboard",
    page_icon="",
    layout="wide",
)

# ---------- Load & cache data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("data/insurance.csv").drop_duplicates().reset_index(drop=True)
    df["smoker_yes"] = (df["smoker"] == "yes").astype(int)
    df["sex_male"]   = (df["sex"]    == "male").astype(int)
    df = pd.get_dummies(df, columns=["region"], drop_first=True, dtype=int)
    df["smoker_bmi"] = df["smoker_yes"] * df["bmi"]
    return df

df = load_data()

st.title(" Medical Insurance Charges — Interactive Dashboard")
#st.caption("M.Sc. Data Science · Statistical Modeling with Python · Lab-4")

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["Data Exploration", "Hypothesis Testing", "Live Prediction"])

# ============================================================
# TAB 1 — Data Exploration
# ============================================================
with tab1:
    st.subheader("Filter the dataset")

    # Sidebar filters
    st.sidebar.header("Filters")

    age_min, age_max = int(df["age"].min()), int(df["age"].max())
    age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))

    bmi_min, bmi_max = float(df["bmi"].min()), float(df["bmi"].max())
    bmi_range = st.sidebar.slider("BMI range", bmi_min, bmi_max, (bmi_min, bmi_max))

    smoker_filter = st.sidebar.multiselect(
        "Smoker", options=["yes", "no"], default=["yes", "no"]
    )

    region_cols = [c for c in df.columns if c.startswith("region_")]
    region_filter = st.sidebar.multiselect(
        "Regions", options=region_cols, default=region_cols
    )

    # Apply filters
    mask = (
        df["age"].between(age_range[0], age_range[1]) &
        df["bmi"].between(bmi_range[0], bmi_range[1]) &
        df["smoker"].isin(smoker_filter)
    )
    for rc in region_cols:
        if rc not in region_filter:
            mask &= (df[rc] == 0)

    filtered = df[mask]

    # Summary stats
    st.write(f"**Rows after filter:** {len(filtered)} / {len(df)}")
    st.dataframe(
        filtered[["age", "bmi", "children", "smoker", "charges"]]
        .describe()
        .round(2)
    )

    # Plot 1: distribution of charges
    fig1 = px.histogram(
        filtered, x="charges", nbins=50,
        title="Distribution of Charges", marginal="box"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Plot 2: scatter age vs charges, colored by smoker
    fig2 = px.scatter(
        filtered, x="age", y="charges", color="smoker",
        title="Age vs Charges (colored by smoker)",
        opacity=0.6
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Plot 3: box of charges by region
    region_long = filtered.melt(
        id_vars=["charges"],
        value_vars=region_cols,
        var_name="region",
        value_name="is_member"
    )
    region_long = region_long[region_long["is_member"] == 1]
    fig3 = px.box(region_long, x="region", y="charges", title="Charges by Region")
    st.plotly_chart(fig3, use_container_width=True)


    # ============================================================
# TAB 2 — Hypothesis Testing Lab
# ============================================================
with tab2:
    st.subheader("Interactive Hypothesis Test")
    st.write("Select a categorical grouping variable and a numeric metric. "
             "The app runs Shapiro-Wilk and Levene's tests, then picks "
             "the correct parametric or non-parametric test automatically.")

    numeric_options = ["charges", "bmi", "age", "children"]
    categorical_options = ["smoker", "sex", "region"]

    col1, col2 = st.columns(2)
    with col1:
        cat_col = st.selectbox("Categorical variable (grouping)", categorical_options)
    with col2:
        num_col = st.selectbox("Numeric metric", numeric_options)

    alpha = st.slider("Significance level (alpha)", 0.01, 0.10, 0.05, 0.01)

    groups = [g[num_col].values for _, g in df.groupby(cat_col)]
    group_names = [str(k) for k, _ in df.groupby(cat_col)]

    # --- Normality per group ---
    st.markdown("**Normality (Shapiro-Wilk) per group**")
    normality_rows = []
    for name, g in zip(group_names, groups):
        stat, p = stats.shapiro(g)
        normality_rows.append({
            "group": name,
            "W": round(stat, 4),
            "p": f"{p:.3e}",
            "normal?": "yes" if p >= alpha else "no"
        })
    st.dataframe(pd.DataFrame(normality_rows), use_container_width=True)

    all_normal = all(float(r["p"]) >= alpha for r in normality_rows)

    # --- Decision ---
    if len(groups) == 2:
        lev_stat, lev_p = stats.levene(*groups)
        st.markdown(f"**Levene's test (equal variance):** stat = {lev_stat:.4f}, p = {lev_p:.3e}")
        equal_var = lev_p >= alpha

        if all_normal and equal_var:
            stat, p = stats.ttest_ind(*groups, equal_var=True)
            test_name = "Two-sample t-test (Student)"
        elif all_normal and not equal_var:
            stat, p = stats.ttest_ind(*groups, equal_var=False)
            test_name = "Two-sample t-test (Welch)"
        else:
            stat, p = stats.mannwhitneyu(*groups, alternative="two-sided")
            test_name = "Mann-Whitney U"

    else:
        lev_stat, lev_p = stats.levene(*groups)
        st.markdown(f"**Levene's test (equal variance):** stat = {lev_stat:.4f}, p = {lev_p:.3e}")

        f_stat, p_anova = stats.f_oneway(*groups)
        h_stat, p_kw = stats.kruskal(*groups)

        st.markdown("**Test results**")
        st.write(f"One-Way ANOVA: F = {f_stat:.4f}, p = {p_anova:.3e}")
        st.write(f"Kruskal-Wallis: H = {h_stat:.4f}, p = {p_kw:.3e}")

        if all_normal and lev_p >= alpha:
            stat, p, test_name = f_stat, p_anova, "One-Way ANOVA"
        else:
            stat, p, test_name = h_stat, p_kw, "Kruskal-Wallis"

    # --- Conclusion ---
    st.markdown("---")
    st.write(f"**Test used:** {test_name}")
    st.write(f"**Statistic:** {stat:.4f}")
    st.write(f"**p-value:** {p:.3e}")
    st.write(f"**alpha:** {alpha}")

    if p < alpha:
        st.success(f"Reject H0 — significant difference in {num_col} across {cat_col}.")
    else:
        st.info(f"Fail to reject H0 — no significant difference in {num_col} across {cat_col}.")

    # --- Group summary ---
    st.markdown("**Group summary**")
    st.dataframe(
        df.groupby(cat_col)[num_col]
          .agg(["count", "mean", "median", "std"])
          .round(2),
        use_container_width=True
    )

    # ============================================================
# Fit final model once (interaction + HC3 robust SEs)
# ============================================================
@st.cache_resource
def fit_final_model():
    dfm = df.copy()
    dfm["log_charges"] = np.log(dfm["charges"])

    X = dfm[["age", "bmi", "children", "smoker_yes", "sex_male",
             "region_northwest", "region_southeast", "region_southwest",
             "smoker_bmi"]]
    X = sm.add_constant(X)
    y = dfm["charges"]

    model = sm.OLS(y, X).fit(cov_type="HC3")
    return model, X

model_final, X_final = fit_final_model()

# ============================================================
# TAB 3 — Live Prediction & Diagnostics
# ============================================================
with tab3:
    st.subheader("Predict medical charges")
    st.write("Adjust the inputs. The model is an OLS regression with a "
             "smoker x bmi interaction term, fitted with HC3 robust standard errors.")

    c1, c2, c3 = st.columns(3)
    with c1:
        in_age = st.slider("Age", 18, 64, 35)
        in_bmi = st.slider("BMI", 15.0, 55.0, 28.0, 0.1)
    with c2:
        in_children = st.slider("Children", 0, 5, 1)
        in_smoker = st.radio("Smoker", ["no", "yes"], horizontal=True)
    with c3:
        in_sex = st.radio("Sex", ["female", "male"], horizontal=True)
        in_region = st.selectbox("Region",
                                 ["northeast", "northwest", "southeast", "southwest"])

    # Build the input row matching X_final columns
    row = {
        "const": 1.0,
        "age": in_age,
        "bmi": in_bmi,
        "children": in_children,
        "smoker_yes": 1 if in_smoker == "yes" else 0,
        "sex_male": 1 if in_sex == "male" else 0,
        "region_northwest": 1 if in_region == "northwest" else 0,
        "region_southeast": 1 if in_region == "southeast" else 0,
        "region_southwest": 1 if in_region == "southwest" else 0,
        "smoker_bmi": (1 if in_smoker == "yes" else 0) * in_bmi,
    }
    X_new = pd.DataFrame([row])[X_final.columns]

    # Point prediction + prediction interval
    pred = model_final.get_prediction(X_new)
    summary = pred.summary_frame(alpha=0.05)

    point = summary["mean"].iloc[0]
    lo_mean = summary["mean_ci_lower"].iloc[0]
    hi_mean = summary["mean_ci_upper"].iloc[0]
    lo_pi = max(0.0, summary["obs_ci_lower"].iloc[0])
    hi_pi = summary["obs_ci_upper"].iloc[0]

    st.markdown("### Predicted charges")
    st.metric("Point estimate", f"${point:,.0f}")
    st.write(f"**95% CI for mean response:** (${lo_mean:,.0f}, ${hi_mean:,.0f})")
    st.write(f"**95% Prediction interval (individual):** (${lo_pi:,.0f}, ${hi_pi:,.0f})")

    # Coefficient table
    st.markdown("### Model coefficients (HC3 robust)")
    coef_df = pd.DataFrame({
        "coef": model_final.params,
        "std_err": model_final.bse,
        "z": model_final.tvalues,
        "p>|z|": model_final.pvalues,
        "CI_low": model_final.conf_int()[0],
        "CI_high": model_final.conf_int()[1],
    }).round(4)
    st.dataframe(coef_df, use_container_width=True)

    # --- Diagnostics ---
    st.markdown("### Residual diagnostics")
    resid = model_final.resid
    fitted = model_final.fittedvalues

    omni_stat, omni_p = omni_normtest(resid)
    jb_stat, jb_p, jb_skew, jb_kurt = jarque_bera(resid)

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("R squared", f"{model_final.rsquared:.3f}")
    d2.metric("Adj R squared", f"{model_final.rsquared_adj:.3f}")
    d3.metric("Skew", f"{jb_skew:.3f}")
    d4.metric("Kurtosis", f"{jb_kurt:.3f}")

    st.write(f"Omnibus: stat = {omni_stat:.3f}, p = {omni_p:.3e}")
    st.write(f"Jarque-Bera: stat = {jb_stat:.3f}, p = {jb_p:.3e}")
    st.write(f"Durbin-Watson: {sm.stats.durbin_watson(resid):.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sm.qqplot(resid, line="45", ax=axes[0], fit=True)
    axes[0].set_title("Q-Q Plot — residuals")

    axes[1].scatter(fitted, resid, alpha=0.5, s=10)
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Fitted charges")
    axes[1].set_ylabel("Residuals")
    axes[1].set_title("Residuals vs Fitted")

    plt.tight_layout()
    st.pyplot(fig)