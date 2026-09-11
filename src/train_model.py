"""
train_model.py
===============
Trains a PREDICTIVE model that answers a genuinely forward-looking question:

    "Based on this vendor's performance THIS quarter, how likely are they
     to breach SLA / have a major disruption NEXT quarter?"

This is different from (and complements) the rule-based score, which
describes CURRENT risk. The ML model tries to anticipate FUTURE risk from
patterns in the data that a simple weighted formula can't easily capture
(e.g. interactions between rising cost variance + slowing corrective
action times + certain categories/regions).

Two models are trained and compared:
    1. Logistic Regression  -> interpretable baseline (coefficients readable)
    2. Random Forest        -> captures non-linear interactions, usually stronger

We also benchmark the RULE-BASED composite score itself as a "classifier"
(using it directly as a risk ranking) so you can honestly say in an
interview: "the ML model beat / matched the business heuristic by X, and
here's why that matters."

Run:
    python src/train_model.py

Outputs:
    models/logistic_regression.pkl
    models/random_forest.pkl
    models/model_comparison.csv
    models/feature_importance.csv
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

DATA_DIR = "data"
MODEL_DIR = "models"

df = pd.read_csv(f"{DATA_DIR}/vendor_quarter_features.csv")
risk_scores = pd.read_csv(f"{DATA_DIR}/vendor_risk_scores.csv")[["vendor_id", "quarter", "composite_risk_score"]]

# Only rows with activity AND a known label (drop the last quarter per vendor,
# which has no "next quarter" to look ahead to)
df = df[df["po_count"] > 0].dropna(subset=["high_risk_next_quarter"]).copy()
df = df.merge(risk_scores, on=["vendor_id", "quarter"], how="left")

# ----------------------------------------------------------------------
# 1. FEATURE SET
#    IMPORTANT: we use only CURRENT-quarter and trailing features — never
#    anything from the future — to avoid data leakage into the prediction.
# ----------------------------------------------------------------------
numeric_features = [
    "on_time_delivery_rate", "avg_delay_days", "delay_days_std", "avg_fill_rate",
    "avg_cost_variance_pct", "cost_variance_std", "expedited_rate",
    "avg_defect_rate", "lot_rejection_rate", "avg_corrective_action_days",
    "disruption_event_count", "disruption_severity_score",
    "credit_health_score", "dpo", "price_volatility",
    "on_time_delivery_rate_trail2q", "avg_defect_rate_trail2q",
    "avg_cost_variance_pct_trail2q", "disruption_severity_score_trail2q",
    "credit_health_score_trail2q", "annual_spend_usd",
]
categorical_features = ["category", "region", "tier"]

for col in numeric_features:
    df[col] = df[col].fillna(df[col].median())

X = df[numeric_features + categorical_features]
y = df["high_risk_next_quarter"].astype(int)

# ----------------------------------------------------------------------
# 2. TRAIN / TEST SPLIT
#    A random split is used here for simplicity/learnability. NOTE (and
#    this is a great interview talking point): in production, a TIME-BASED
#    split (train on earlier quarters, test on later ones) is more rigorous
#    for a forecasting problem, since it avoids "seeing the future" during
#    training. We do both, and report both, below.
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.25, random_state=42, stratify=y
)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

log_reg = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
])

rand_forest = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    )),
])

log_reg.fit(X_train, y_train)
rand_forest.fit(X_train, y_train)

# ----------------------------------------------------------------------
# 3. TIME-BASED SPLIT (more rigorous, mentioned above)
# ----------------------------------------------------------------------
quarters_sorted = sorted(df["quarter"].unique())
cutoff = quarters_sorted[int(len(quarters_sorted) * 0.75)]
time_train_mask = df["quarter"] < cutoff
time_test_mask = df["quarter"] >= cutoff

Xt_train, yt_train = X[time_train_mask], y[time_train_mask]
Xt_test, yt_test = X[time_test_mask], y[time_test_mask]

rand_forest_time = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    )),
])
rand_forest_time.fit(Xt_train, yt_train)

# ----------------------------------------------------------------------
# 4. EVALUATE ALL APPROACHES
# ----------------------------------------------------------------------
def evaluate(name, y_true, y_pred_proba, y_pred_label):
    return {
        "model": name,
        "roc_auc": round(roc_auc_score(y_true, y_pred_proba), 4),
        "precision": round(precision_score(y_true, y_pred_label, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred_label, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred_label, zero_division=0), 4),
    }

results = []

# Logistic Regression (random split)
proba = log_reg.predict_proba(X_test)[:, 1]
results.append(evaluate("Logistic Regression (random split)", y_test, proba, log_reg.predict(X_test)))

# Random Forest (random split)
proba = rand_forest.predict_proba(X_test)[:, 1]
results.append(evaluate("Random Forest (random split)", y_test, proba, rand_forest.predict(X_test)))

# Random Forest (TIME-BASED split -- more realistic)
proba_t = rand_forest_time.predict_proba(Xt_test)[:, 1]
results.append(evaluate("Random Forest (time-based split)", yt_test, proba_t, rand_forest_time.predict(Xt_test)))

# Rule-based composite score used AS-IS as a risk ranking (benchmark)
rule_scores_test = df.loc[idx_test, "composite_risk_score"].fillna(df["composite_risk_score"].median())
rule_auc = roc_auc_score(y_test, rule_scores_test)
rule_pred_label = (rule_scores_test >= rule_scores_test.median()).astype(int)
results.append({
    "model": "Rule-based composite score (benchmark)",
    "roc_auc": round(rule_auc, 4),
    "precision": round(precision_score(y_test, rule_pred_label, zero_division=0), 4),
    "recall": round(recall_score(y_test, rule_pred_label, zero_division=0), 4),
    "f1": round(f1_score(y_test, rule_pred_label, zero_division=0), 4),
})

comparison_df = pd.DataFrame(results)
comparison_df.to_csv(f"{MODEL_DIR}/model_comparison.csv", index=False)

# ----------------------------------------------------------------------
# 5. FEATURE IMPORTANCE (Random Forest, random-split model)
# ----------------------------------------------------------------------
ohe = rand_forest.named_steps["prep"].named_transformers_["cat"]
cat_names = list(ohe.get_feature_names_out(categorical_features))
all_feature_names = numeric_features + cat_names

importances = rand_forest.named_steps["clf"].feature_importances_
feat_imp_df = pd.DataFrame({
    "feature": all_feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)
feat_imp_df.to_csv(f"{MODEL_DIR}/feature_importance.csv", index=False)

# ----------------------------------------------------------------------
# 6. SAVE MODELS
# ----------------------------------------------------------------------
joblib.dump(log_reg, f"{MODEL_DIR}/logistic_regression.pkl")
joblib.dump(rand_forest, f"{MODEL_DIR}/random_forest.pkl")

# Also save the test-set predictions for dashboard visualization
test_output = df.loc[idx_test, ["vendor_id", "quarter", "category", "region"]].copy()
test_output["actual_high_risk"] = y_test.values
test_output["rf_predicted_probability"] = rand_forest.predict_proba(X_test)[:, 1]
test_output["rule_based_score"] = rule_scores_test.values
test_output.to_csv(f"{MODEL_DIR}/test_predictions.csv", index=False)

print("Model training complete.\n")
print(comparison_df.to_string(index=False))
print("\nTop 10 predictive features (Random Forest):")
print(feat_imp_df.head(10).to_string(index=False))
