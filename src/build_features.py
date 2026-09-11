"""
build_features.py
==================
Turns the raw transactional tables (POs, inspections, disruptions, financial
signals) into a VENDOR-QUARTER feature table. This is the single most
important analytical step in the project: it's where messy operational
data becomes structured signals that a risk score or ML model can use.

Grain: one row = one vendor, one quarter (e.g. V0001, 2023Q3).

Run:
    python src/build_features.py

Output:
    data/vendor_quarter_features.csv
"""

import pandas as pd
import numpy as np

DATA_DIR = "data"

vendors = pd.read_csv(f"{DATA_DIR}/vendors.csv")
po = pd.read_csv(f"{DATA_DIR}/purchase_orders.csv", parse_dates=["order_date", "promised_delivery_date", "actual_delivery_date"])
qi = pd.read_csv(f"{DATA_DIR}/quality_inspections.csv", parse_dates=["inspection_date"])
events = pd.read_csv(f"{DATA_DIR}/disruption_events.csv", parse_dates=["event_date"])
fin = pd.read_csv(f"{DATA_DIR}/vendor_financial_signals.csv")

po["quarter"] = po["actual_delivery_date"].dt.to_period("Q").astype(str)
qi["quarter"] = qi["inspection_date"].dt.to_period("Q").astype(str)
events["quarter"] = events["event_date"].dt.to_period("Q").astype(str)

# ----------------------------------------------------------------------
# 1. DELIVERY / OTD FEATURES (from purchase_orders)
# ----------------------------------------------------------------------
po["on_time"] = po["delay_days"] <= 2  # small grace window, common in real OTD definitions
po["fill_rate"] = po["quantity_received"] / po["quantity_ordered"]
po["cost_variance_pct"] = (po["actual_unit_cost_usd"] - po["quoted_unit_cost_usd"]) / po["quoted_unit_cost_usd"]

delivery_agg = po.groupby(["vendor_id", "quarter"]).agg(
    po_count=("po_id", "count"),
    on_time_delivery_rate=("on_time", "mean"),
    avg_delay_days=("delay_days", "mean"),
    delay_days_std=("delay_days", "std"),
    avg_fill_rate=("fill_rate", "mean"),
    avg_cost_variance_pct=("cost_variance_pct", "mean"),
    cost_variance_std=("cost_variance_pct", "std"),
    total_spend_usd=("actual_unit_cost_usd", lambda s: (s * po.loc[s.index, "quantity_received"]).sum()),
    expedited_rate=("expedited_flag", "mean"),
).reset_index()

# ----------------------------------------------------------------------
# 2. QUALITY FEATURES (from quality_inspections)
# ----------------------------------------------------------------------
quality_agg = qi.groupby(["vendor_id", "quarter"]).agg(
    inspection_count=("inspection_id", "count"),
    avg_defect_rate=("defect_rate", "mean"),
    lot_rejection_rate=("lot_rejected_flag", "mean"),
    avg_corrective_action_days=("corrective_action_days", "mean"),
).reset_index()

# ----------------------------------------------------------------------
# 3. DISRUPTION EVENT FEATURES
# ----------------------------------------------------------------------
severity_weight = {"Low": 1, "Medium": 2, "High": 4, "Critical": 8}
events["severity_weight"] = events["severity"].map(severity_weight)

event_agg = events.groupby(["vendor_id", "quarter"]).agg(
    disruption_event_count=("event_id", "count"),
    disruption_severity_score=("severity_weight", "sum"),
    max_resolution_days=("resolution_days", "max"),
).reset_index()

# ----------------------------------------------------------------------
# 4. FINANCIAL SIGNAL FEATURES (already at vendor-quarter grain)
# ----------------------------------------------------------------------
fin_features = fin.rename(columns={
    "credit_risk_score_proxy": "credit_health_score",
    "days_payable_outstanding": "dpo",
    "price_volatility_index": "price_volatility",
})

# ----------------------------------------------------------------------
# 5. BUILD FULL VENDOR x QUARTER SCAFFOLD & MERGE EVERYTHING
# ----------------------------------------------------------------------
all_quarters = sorted(fin_features["quarter"].unique())
scaffold = pd.MultiIndex.from_product(
    [vendors["vendor_id"], all_quarters], names=["vendor_id", "quarter"]
).to_frame(index=False)

features = scaffold.merge(delivery_agg, on=["vendor_id", "quarter"], how="left")
features = features.merge(quality_agg, on=["vendor_id", "quarter"], how="left")
features = features.merge(event_agg, on=["vendor_id", "quarter"], how="left")
features = features.merge(fin_features, on=["vendor_id", "quarter"], how="left")
features = features.merge(
    vendors[["vendor_id", "category", "country", "region", "tier", "annual_spend_usd",
             "single_source_flag", "certifications"]],
    on="vendor_id", how="left"
)

# Quarters with no PO activity -> fill with neutral/zero values (no orders = no delay/defect signal)
fill_zero = ["po_count", "on_time_delivery_rate", "avg_delay_days", "delay_days_std",
             "avg_fill_rate", "avg_cost_variance_pct", "cost_variance_std", "total_spend_usd",
             "expedited_rate", "inspection_count", "avg_defect_rate", "lot_rejection_rate",
             "avg_corrective_action_days", "disruption_event_count", "disruption_severity_score",
             "max_resolution_days"]
for col in fill_zero:
    features[col] = features[col].fillna(0)

features["avg_fill_rate"] = features["avg_fill_rate"].replace(0, np.nan)  # don't punish "no orders" as bad fill rate
features["on_time_delivery_rate"] = features.apply(
    lambda r: r["on_time_delivery_rate"] if r["po_count"] > 0 else np.nan, axis=1
)

# ----------------------------------------------------------------------
# 6. TRAILING (ROLLING) FEATURES — critical for a model that PREDICTS
#    forward risk instead of just describing the past. We compute a
#    trailing 2-quarter average per vendor for key signals.
# ----------------------------------------------------------------------
features = features.sort_values(["vendor_id", "quarter"])

rolling_cols = ["on_time_delivery_rate", "avg_defect_rate", "avg_cost_variance_pct",
                 "disruption_severity_score", "credit_health_score"]

for col in rolling_cols:
    features[f"{col}_trail2q"] = (
        features.groupby("vendor_id")[col]
        .transform(lambda s: s.rolling(window=2, min_periods=1).mean())
    )

# ----------------------------------------------------------------------
# 7. TARGET LABEL FOR ML: "high_risk_next_quarter"
#    Defined as: in the FOLLOWING quarter, the vendor either
#      (a) had on-time delivery rate < 0.75, OR
#      (b) had a disruption event with severity >= High, OR
#      (c) had lot rejection rate > 0.15
#    This mirrors how real vendor-risk platforms (e.g. Resilinc,
#    riskmethods) define an "SLA breach / disruption" event.
# ----------------------------------------------------------------------
features["breach_this_quarter"] = (
    (features["on_time_delivery_rate"] < 0.75) |
    (features["disruption_severity_score"] >= 4) |
    (features["lot_rejection_rate"] > 0.15)
).astype(int)

features = features.sort_values(["vendor_id", "quarter"])
features["high_risk_next_quarter"] = (
    features.groupby("vendor_id")["breach_this_quarter"].shift(-1)
)

features.to_csv(f"{DATA_DIR}/vendor_quarter_features.csv", index=False)

print("Feature table built.")
print(f"  Shape: {features.shape}")
print(f"  Quarters covered: {features['quarter'].min()} to {features['quarter'].max()}")
print(f"  Positive class rate (high_risk_next_quarter): "
      f"{features['high_risk_next_quarter'].dropna().mean():.2%}")
