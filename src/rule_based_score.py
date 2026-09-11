"""
rule_based_score.py
====================
Builds a TRANSPARENT, weighted composite Vendor Risk Score (0-100, higher =
riskier). This is the kind of scorecard a procurement/vendor-risk team can
actually understand and defend to a vendor in a business review — no black
box. It mirrors real-world vendor scorecarding frameworks used in SRM tools.

Methodology: Min-max normalize each raw signal to a 0-100 "risk sub-score"
(higher = worse), then combine with business-assigned weights.

Weights (must sum to 1.0):
    Delivery reliability (OTD)         25%
    Quality (defect/rejection rate)    20%
    Cost discipline (cost variance)    15%
    Lead-time consistency (variability)15%
    Disruption exposure                15%
    Financial health                   10%

Concentration risk (single-source + high spend) is applied as a separate
risk MULTIPLIER/flag rather than folded into the weighted average, because
in practice it's a structural risk (dependency), not a performance metric.

Run:
    python src/rule_based_score.py

Output:
    data/vendor_risk_scores.csv
"""

import pandas as pd
import numpy as np

DATA_DIR = "data"

df = pd.read_csv(f"{DATA_DIR}/vendor_quarter_features.csv")

WEIGHTS = {
    "otd_risk": 0.25,
    "quality_risk": 0.20,
    "cost_risk": 0.15,
    "leadtime_risk": 0.15,
    "disruption_risk": 0.15,
    "financial_risk": 0.10,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def minmax_risk(series, invert=False, clip_quantile=0.98):
    """
    Normalize a raw signal to a 0-100 RISK sub-score.
    invert=True means higher raw value = LOWER risk (e.g. on-time delivery rate),
    so we flip the scale.
    Clips extreme outliers at the 98th percentile so one freak event doesn't
    dominate the whole score (a very deliberate, explainable design choice).
    """
    s = series.copy()
    upper = s.quantile(clip_quantile)
    s = s.clip(upper=upper)
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-9:
        return pd.Series(50, index=series.index)  # no variance -> neutral score
    norm = (s - lo) / (hi - lo) * 100
    return (100 - norm) if invert else norm


# Only score quarters where the vendor actually had activity (po_count > 0)
scored = df[df["po_count"] > 0].copy()

# --- Sub-scores (each 0-100, higher = riskier) ---
scored["otd_risk"] = minmax_risk(scored["on_time_delivery_rate"].fillna(scored["on_time_delivery_rate"].median()), invert=True)
scored["quality_risk"] = minmax_risk(scored["avg_defect_rate"].fillna(0))
scored["cost_risk"] = minmax_risk(scored["avg_cost_variance_pct"].abs().fillna(0))
scored["leadtime_risk"] = minmax_risk(scored["delay_days_std"].fillna(0))
scored["disruption_risk"] = minmax_risk(scored["disruption_severity_score"].fillna(0))
scored["financial_risk"] = minmax_risk(scored["credit_health_score"].fillna(scored["credit_health_score"].median()), invert=True)

# --- Weighted composite ---
scored["composite_risk_score"] = sum(scored[k] * w for k, w in WEIGHTS.items())

# --- Concentration risk flag & multiplier ---
high_spend_threshold = scored["annual_spend_usd"].quantile(0.75)
scored["concentration_risk_flag"] = (
    scored["single_source_flag"] & (scored["annual_spend_usd"] >= high_spend_threshold)
)
# A flagged vendor gets a bounded uplift (not unbounded) — reflects "this
# vendor's failure would hurt a lot more" without letting it swamp the score.
scored["composite_risk_score"] = np.where(
    scored["concentration_risk_flag"],
    np.minimum(100, scored["composite_risk_score"] * 1.15),
    scored["composite_risk_score"]
)

# --- Risk tier for business communication ---
def tier_label(score):
    if score >= 70:
        return "Critical"
    elif score >= 50:
        return "High"
    elif score >= 30:
        return "Moderate"
    else:
        return "Low"

scored["risk_tier"] = scored["composite_risk_score"].apply(tier_label)

output_cols = [
    "vendor_id", "quarter", "category", "country", "region", "tier",
    "annual_spend_usd", "single_source_flag", "concentration_risk_flag",
    "on_time_delivery_rate", "avg_defect_rate", "avg_cost_variance_pct",
    "delay_days_std", "disruption_severity_score", "credit_health_score",
    "otd_risk", "quality_risk", "cost_risk", "leadtime_risk",
    "disruption_risk", "financial_risk",
    "composite_risk_score", "risk_tier",
    "high_risk_next_quarter",
]
result = scored[output_cols].round(2)
result.to_csv(f"{DATA_DIR}/vendor_risk_scores.csv", index=False)

print("Rule-based risk scoring complete.")
print(result["risk_tier"].value_counts())
print()
print("Sample high-risk vendors (latest quarter):")
latest_q = result["quarter"].max()
print(result[result["quarter"] == latest_q].sort_values("composite_risk_score", ascending=False)
      [["vendor_id", "category", "country", "composite_risk_score", "risk_tier"]].head(10).to_string(index=False))
