"""
app.py
======
Interactive Streamlit dashboard for the Vendor Risk & Supply Chain
Diagnostics project. Run from the project root with:

    streamlit run dashboard/app.py

Pages:
    1. Executive Overview   - KPIs, risk heatmap, trend
    2. Vendor Scorecard     - drill into a single vendor
    3. Predictive Risk Model- ML vs rule-based comparison, feature importance
    4. Alerts & Watchlist   - vendors flagged for review this quarter
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Vendor Risk & Supply Chain Diagnostics", layout="wide", page_icon="📦")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
MODEL_DIR = os.path.join(BASE, "models")

# ----------------------------------------------------------------------
# DATA LOADING (cached)
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    vendors = pd.read_csv(f"{DATA_DIR}/vendors.csv")
    risk_scores = pd.read_csv(f"{DATA_DIR}/vendor_risk_scores.csv")
    features = pd.read_csv(f"{DATA_DIR}/vendor_quarter_features.csv")
    events = pd.read_csv(f"{DATA_DIR}/disruption_events.csv", parse_dates=["event_date"])
    po = pd.read_csv(f"{DATA_DIR}/purchase_orders.csv", parse_dates=["order_date", "actual_delivery_date"])
    return vendors, risk_scores, features, events, po

@st.cache_data
def load_model_outputs():
    comparison = pd.read_csv(f"{MODEL_DIR}/model_comparison.csv")
    feat_imp = pd.read_csv(f"{MODEL_DIR}/feature_importance.csv")
    test_pred = pd.read_csv(f"{MODEL_DIR}/test_predictions.csv")
    return comparison, feat_imp, test_pred

vendors, risk_scores, features, events, po = load_data()
model_comparison, feat_importance, test_predictions = load_model_outputs()

quarters = sorted(risk_scores["quarter"].unique())
latest_q = quarters[-1]

# ----------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------
st.sidebar.title("📦 Vendor Risk Diagnostics")
page = st.sidebar.radio("Navigate", [
    "Executive Overview",
    "Vendor Scorecard",
    "Predictive Risk Model",
    "Alerts & Watchlist",
    "About this Project",
])
selected_quarter = st.sidebar.selectbox("Reporting Quarter", quarters, index=len(quarters) - 1)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Synthetic data simulating an electronics/hardware manufacturer's "
    "global vendor base (2022–2026). Built for portfolio/interview demonstration."
)

# ========================================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ========================================================================
if page == "Executive Overview":
    st.title("Executive Overview — Vendor Risk Posture")
    st.caption(f"Reporting period: **{selected_quarter}**")

    q_data = risk_scores[risk_scores["quarter"] == selected_quarter]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Vendors", f"{q_data['vendor_id'].nunique()}")
    col2.metric("High / Critical Risk", f"{(q_data['risk_tier'].isin(['High','Critical'])).sum()}",
                delta=None)
    avg_otd = q_data["on_time_delivery_rate"].mean()
    col3.metric("Avg On-Time Delivery", f"{avg_otd:.1%}")
    disrupt_count = events[events["event_date"].dt.to_period("Q").astype(str) == selected_quarter].shape[0]
    col4.metric("Disruption Events (Qtr)", disrupt_count)

    st.markdown("---")

    c1, c2 = st.columns([1.3, 1])

    with c1:
        st.subheader("Risk Heatmap — Category × Region")
        heat = q_data.groupby(["category", "region"])["composite_risk_score"].mean().reset_index()
        heat_pivot = heat.pivot(index="category", columns="region", values="composite_risk_score")
        fig = px.imshow(
            heat_pivot, text_auto=".0f", color_continuous_scale="RdYlGn_r",
            aspect="auto", labels=dict(color="Avg Risk Score")
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Risk Tier Distribution")
        tier_counts = q_data["risk_tier"].value_counts().reindex(["Low", "Moderate", "High", "Critical"]).fillna(0)
        colors = {"Low": "#2ca02c", "Moderate": "#ffbb33", "High": "#ff7043", "Critical": "#d32f2f"}
        fig2 = go.Figure(go.Bar(
            x=tier_counts.index, y=tier_counts.values,
            marker_color=[colors[t] for t in tier_counts.index]
        ))
        fig2.update_layout(height=420, xaxis_title="", yaxis_title="Vendor Count")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Portfolio Risk Trend Over Time")
    trend = risk_scores.groupby("quarter").agg(
        avg_risk_score=("composite_risk_score", "mean"),
        pct_high_risk=("risk_tier", lambda s: s.isin(["High", "Critical"]).mean() * 100)
    ).reset_index()
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=trend["quarter"], y=trend["avg_risk_score"], name="Avg Risk Score",
                               mode="lines+markers", line=dict(color="#1f77b4", width=3)))
    fig3.add_trace(go.Scatter(x=trend["quarter"], y=trend["pct_high_risk"], name="% Vendors High/Critical",
                               mode="lines+markers", line=dict(color="#d32f2f", width=2, dash="dot"), yaxis="y2"))
    fig3.update_layout(
        height=380,
        yaxis=dict(title="Avg Composite Risk Score"),
        yaxis2=dict(title="% High/Critical", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(
        "Notice the risk spikes aligning with real-world-style macro shock windows built into "
        "the simulation (e.g. chip shortage tail in 2022, Red Sea shipping disruption late 2023, "
        "tariff escalation in 2025). These spikes are driven by dips in on-time delivery and "
        "rises in cost variance during those windows — not by the discrete disruption event log, "
        "which is a separate, less time-correlated signal. This is intentional so you can discuss "
        "how macro shocks propagate into operational KPIs."
    )

# ========================================================================
# PAGE 2: VENDOR SCORECARD
# ========================================================================
elif page == "Vendor Scorecard":
    st.title("Vendor Scorecard")

    vendor_list = sorted(vendors["vendor_id"].unique())
    default_idx = 0
    vendor_id = st.selectbox("Select Vendor", vendor_list, index=default_idx)

    v_info = vendors[vendors["vendor_id"] == vendor_id].iloc[0]
    v_scores = risk_scores[risk_scores["vendor_id"] == vendor_id].sort_values("quarter")

    st.subheader(f"{v_info['vendor_name']}  ·  {v_info['category']}  ·  {v_info['country']}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tier", v_info["tier"])
    c2.metric("Annual Spend", f"${v_info['annual_spend_usd']:,.0f}")
    c3.metric("Single-Source", "⚠️ Yes" if v_info["single_source_flag"] else "No")
    latest_row = v_scores[v_scores["quarter"] == v_scores["quarter"].max()]
    latest_score = latest_row["composite_risk_score"].values[0] if len(latest_row) else np.nan
    c4.metric("Current Risk Score", f"{latest_score:.1f}" if not np.isnan(latest_score) else "N/A")

    st.markdown("---")
    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.subheader("Risk Score Trend")
        fig = px.line(v_scores, x="quarter", y="composite_risk_score", markers=True)
        fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.08, line_width=0)
        fig.add_hrect(y0=50, y1=70, fillcolor="orange", opacity=0.08, line_width=0)
        fig.update_layout(height=350, yaxis_title="Composite Risk Score (0-100)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Latest Sub-Score Breakdown")
        if len(latest_row):
            radar_cats = ["otd_risk", "quality_risk", "cost_risk", "leadtime_risk", "disruption_risk", "financial_risk"]
            radar_vals = latest_row[radar_cats].values.flatten().tolist()
            radar_vals.append(radar_vals[0])
            radar_labels = ["On-Time Delivery", "Quality", "Cost Discipline", "Lead-Time Consistency",
                             "Disruption Exposure", "Financial Health"]
            radar_labels.append(radar_labels[0])
            fig_r = go.Figure(go.Scatterpolar(r=radar_vals, theta=radar_labels, fill="toself"))
            fig_r.update_layout(height=350, polar=dict(radialaxis=dict(range=[0, 100])), showlegend=False)
            st.plotly_chart(fig_r, use_container_width=True)
        else:
            st.info("No scored quarters for this vendor yet.")

    st.markdown("---")
    st.subheader("Operational Detail")
    v_po = po[po["vendor_id"] == vendor_id].sort_values("order_date")
    if len(v_po):
        tab1, tab2 = st.tabs(["Delivery Timeliness", "Recent Purchase Orders"])
        with tab1:
            fig_d = px.scatter(v_po, x="order_date", y="delay_days", color="delay_days",
                                color_continuous_scale="RdYlGn_r",
                                labels={"delay_days": "Delay (days)", "order_date": "Order Date"})
            fig_d.add_hline(y=0, line_dash="dot")
            st.plotly_chart(fig_d, use_container_width=True)
        with tab2:
            st.dataframe(
                v_po[["po_id", "order_date", "promised_delivery_date", "actual_delivery_date",
                      "delay_days", "quantity_ordered", "quantity_received", "actual_unit_cost_usd"]]
                .sort_values("order_date", ascending=False).head(25),
                use_container_width=True, hide_index=True
            )
    else:
        st.info("No purchase order history for this vendor.")

# ========================================================================
# PAGE 3: PREDICTIVE RISK MODEL
# ========================================================================
elif page == "Predictive Risk Model":
    st.title("Predictive Risk Model — ML vs. Rule-Based Scorecard")
    st.markdown(
        "This page compares the transparent **rule-based composite score** against a "
        "**trained machine learning model** that predicts whether a vendor will breach "
        "SLA / have a major disruption **next quarter**, using only current and trailing "
        "performance signals (no future information — leakage-safe by design)."
    )

    st.subheader("Model Performance Comparison")
    st.dataframe(model_comparison.style.background_gradient(subset=["roc_auc", "f1"], cmap="RdYlGn"),
                 use_container_width=True, hide_index=True)
    st.caption(
        "ROC-AUC around 0.70–0.75 here is realistic and expected for this kind of noisy, "
        "operational forecasting problem — a perfect score would actually be a red flag for "
        "data leakage. The Random Forest modestly outperforms the rule-based benchmark, which "
        "is the expected result: rules are transparent and explainable, ML captures non-linear "
        "interactions the rules can't."
    )

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Predictive Features (Random Forest)")
        top_feat = feat_importance.head(12)
        fig = px.bar(top_feat.sort_values("importance"), x="importance", y="feature", orientation="h")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Predicted Risk Probability Distribution")
        fig2 = px.histogram(test_predictions, x="rf_predicted_probability", color="actual_high_risk",
                             nbins=25, barmode="overlay", opacity=0.65,
                             labels={"actual_high_risk": "Actually Breached Next Qtr"})
        fig2.update_layout(height=450)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(
            "Good separation between the two distributions means the model is finding real "
            "signal — vendors it scores as high-probability are genuinely more likely to breach."
        )

    st.markdown("---")
    st.subheader("Model vs. Rule-Based Score — Same Vendor, Same Quarter")
    merged = test_predictions.copy()
    fig3 = px.scatter(
        merged, x="rule_based_score", y="rf_predicted_probability", color="actual_high_risk",
        hover_data=["vendor_id", "quarter"],
        labels={"rule_based_score": "Rule-Based Risk Score (0-100)",
                "rf_predicted_probability": "ML-Predicted Breach Probability"}
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(
        "Points where the two methods disagree (e.g. low rule-based score but high ML "
        "probability) are the most interesting for a real team to investigate — the ML model "
        "may be picking up an early warning pattern the static rule-based formula misses."
    )

# ========================================================================
# PAGE 4: ALERTS & WATCHLIST
# ========================================================================
elif page == "Alerts & Watchlist":
    st.title("Alerts & Watchlist")
    st.caption(f"Vendors flagged for review — {selected_quarter}")

    q_data = risk_scores[risk_scores["quarter"] == selected_quarter].copy()
    watchlist = q_data[q_data["risk_tier"].isin(["High", "Critical"])].sort_values(
        "composite_risk_score", ascending=False
    )

    st.markdown(f"**{len(watchlist)} vendors** flagged High or Critical risk this quarter.")

    if len(watchlist):
        display_cols = ["vendor_id", "category", "country", "tier", "annual_spend_usd",
                         "concentration_risk_flag", "on_time_delivery_rate", "avg_defect_rate",
                         "composite_risk_score", "risk_tier"]
        st.dataframe(
            watchlist[display_cols].rename(columns={
                "annual_spend_usd": "Annual Spend ($)",
                "concentration_risk_flag": "Concentration Risk",
                "on_time_delivery_rate": "OTD Rate",
                "avg_defect_rate": "Defect Rate",
                "composite_risk_score": "Risk Score",
                "risk_tier": "Tier",
            }),
            use_container_width=True, hide_index=True
        )

        st.markdown("---")
        st.subheader("Suggested Actions (business logic, not ML-generated)")
        for _, row in watchlist.head(5).iterrows():
            reasons = []
            if row["on_time_delivery_rate"] < 0.75:
                reasons.append("chronic late delivery")
            if row["avg_defect_rate"] > 0.03:
                reasons.append("elevated defect rate")
            if row["concentration_risk_flag"]:
                reasons.append("single-source concentration exposure")
            reason_text = ", ".join(reasons) if reasons else "composite score threshold breach"
            st.markdown(f"- **{row['vendor_id']}** ({row['category']}, {row['country']}): "
                         f"flagged for *{reason_text}*. Recommend supplier business review "
                         f"{'and dual-sourcing evaluation' if row['concentration_risk_flag'] else ''}.")
    else:
        st.success("No vendors currently flagged High or Critical risk.")

# ========================================================================
# PAGE 5: ABOUT
# ========================================================================
elif page == "About this Project":
    st.title("About This Project")
    st.markdown("""
### Enterprise Supply Chain Operational Risk & Vendor Performance Diagnostics

**Problem simulated:** A global electronics/hardware manufacturer sources components
(semiconductors, PCBs, displays, connectors, batteries, enclosures, packaging) from
~70 vendors across APAC, EMEA, and the Americas. Procurement and supply chain teams
need to know, at any point in time: *which vendors are operationally risky right now,
and which are likely to become risky next quarter* — before a line-down event happens.

**Why this isn't a generic Kaggle project:** There is no public dataset for internal
vendor performance and risk data (it's confidential to every real company that has it).
This project simulates the data-generation process realistically — including
time-anchored macro disruption events modeled on real 2022–2026 supply chain shocks
(chip shortage tail, regional lockdowns, Red Sea shipping disruption, tariff
escalation) — rather than reusing an existing static dataset.

**Two-layer risk methodology:**
1. **Rule-based composite scorecard** — a transparent, weighted formula
   (delivery 25%, quality 20%, cost 15%, lead-time consistency 15%,
   disruption exposure 15%, financial health 10%, plus a concentration-risk
   multiplier for single-source high-spend vendors). Fully explainable to
   a non-technical stakeholder.
2. **Predictive ML model** (Logistic Regression + Random Forest) — forecasts
   which vendors are likely to breach SLA next quarter, using only
   current/trailing signals (leakage-safe), benchmarked directly against
   the rule-based score.

**Tech stack:** Python, pandas, NumPy, scikit-learn, Streamlit, Plotly.

See `docs/README.md` in the project files for the full methodology write-up
and interview preparation notes.
""")
