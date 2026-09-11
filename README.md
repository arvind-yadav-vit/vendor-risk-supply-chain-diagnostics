# Enterprise Supply Chain Operational Risk & Vendor Performance Diagnostics

A vendor risk monitoring and prediction system for a simulated global electronics/hardware
manufacturer — built to demonstrate real supply-chain analytics skills: data modeling,
business-rule scorecarding, predictive ML, and stakeholder-facing dashboarding.

---

## 1. The Business Problem

A hardware manufacturer sources components — semiconductors, PCBs, displays, connectors,
batteries, enclosures, packaging — from ~70 vendors spread across Asia, Europe, and the
Americas. Procurement and supply chain leadership need to answer two questions continuously:

1. **Which vendors are operationally risky *right now*?** (late deliveries, quality issues,
   cost overruns, single-source dependency)
2. **Which vendors are likely to *become* risky next quarter?** — so the team can act
   *before* a stockout or line-down event, not after.

This project builds both: a transparent scorecard for question 1, and a predictive model
for question 2.

## 2. Why This Isn't a Generic Kaggle Project

There's no public dataset of confidential internal vendor performance data — no real
company publishes their supplier scorecards. So instead of reusing a static, overused
dataset, this project **simulates the data-generation process** the way it would actually
happen inside an ERP/SRM system (SAP Ariba / Coupa-style), including:

- Realistic entity relationships (vendors → purchase orders → quality inspections →
  disruption events → financial signals)
- **Time-anchored macro disruption events** modeled on real 2022–2026 supply chain shocks:
  the 2022 global chip shortage tail, China regional lockdowns, the late-2023 Red Sea
  shipping disruption, 2024 Taiwan seismic/grid strain, and 2025–2026 tariff escalation
- Archetype-driven vendor behavior (some vendors are structurally reliable, some are
  structurally risky) so the data has genuine, learnable signal — not random noise

## 3. Project Architecture

```
vendor-risk-project/
├── src/
│   ├── generate_data.py       # Step 1: simulate raw transactional data
│   ├── build_features.py      # Step 2: raw data -> vendor-quarter feature table
│   ├── rule_based_score.py    # Step 3: transparent weighted risk scorecard
│   ├── train_model.py         # Step 4: predictive ML model + benchmark comparison
│   └── make_notebook.py       # generates the EDA notebook
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
├── data/                      # generated CSVs (raw + feature tables + scores)
├── models/                    # trained model files + evaluation outputs
├── dashboard/
│   └── app.py                 # Streamlit interactive dashboard
├── run_pipeline.py            # runs steps 1-4 end to end
├── requirements.txt
└── README.md                  # this file
```

### Data flow (how the pieces connect)

```
generate_data.py
   → vendors.csv, purchase_orders.csv, quality_inspections.csv,
     disruption_events.csv, vendor_financial_signals.csv
        ↓
build_features.py   (aggregates transactions to vendor × quarter grain,
                      builds trailing/rolling features, defines the
                      "high_risk_next_quarter" prediction target)
        ↓
   vendor_quarter_features.csv
        ↓
   ┌────────────────────┬──────────────────────┐
   ↓                                            ↓
rule_based_score.py                      train_model.py
(transparent weighted scorecard)         (Logistic Regression + Random Forest,
   ↓                                       benchmarked against the rule-based score)
vendor_risk_scores.csv                          ↓
   ↓                                      models/*.pkl, model_comparison.csv
   └──────────────────┬───────────────────┘
                       ↓
              dashboard/app.py (Streamlit)
```

## 4. How To Run It

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full data + modeling pipeline
python run_pipeline.py

# 3. Launch the interactive dashboard
streamlit run dashboard/app.py
```

Everything is seeded (`SEED = 42`), so re-running `run_pipeline.py` reproduces identical
results — an important, deliberate detail for a project you want to demo reliably.

## 5. Methodology Deep-Dive

### 5.1 Rule-based composite risk score (the transparent layer)

Each vendor-quarter gets six sub-scores (0–100, higher = riskier), each built by
min-max normalizing a raw operational signal (with outlier clipping at the 98th
percentile so one freak event can't dominate the score):

| Sub-score | Signal | Weight |
|---|---|---|
| On-time delivery risk | On-time delivery rate (inverted) | 25% |
| Quality risk | Average defect rate | 20% |
| Cost risk | Absolute cost variance vs. quoted price | 15% |
| Lead-time consistency risk | Std. deviation of delivery delay | 15% |
| Disruption exposure risk | Severity-weighted disruption event count | 15% |
| Financial health risk | Credit health score proxy (inverted) | 10% |

These combine into a **weighted composite score**, then a **concentration risk
multiplier** (+15%, capped at 100) is applied to vendors that are both single-source
*and* in the top quartile of spend — because losing that vendor would hurt
disproportionately, which is a structural risk, not a performance metric, so it's
handled separately rather than averaged in.

**Why weights, not equal averaging?** In a real procurement org, delivery reliability
and quality failures cause immediate production impact, so they're weighted higher
than, say, cost variance, which is painful but rarely stops a production line.

### 5.2 Predictive ML model (the forward-looking layer)

The rule-based score describes **current** risk. The ML model tries to answer a
harder, more valuable question: *"given this vendor's behavior this quarter, how
likely are they to breach SLA or have a major disruption **next** quarter?"*

- **Target:** `high_risk_next_quarter` — a vendor-quarter is labeled 1 if, in the
  *following* quarter, the vendor has OTD < 75%, OR a disruption event of severity
  High/Critical, OR a lot rejection rate > 15%.
- **Leakage-safe by design:** the target uses only *future* information for the
  label, and the model only sees *current and trailing* features — never data from
  the quarter being predicted.
- **Models trained:** Logistic Regression (interpretable baseline) and Random Forest
  (captures non-linear interactions). Both evaluated with a random train/test split
  *and* a time-based split (train on earlier quarters, test on later ones — the more
  rigorous approach for a forecasting problem, included specifically so you can
  discuss the trade-off in an interview).
- **Benchmarked directly against the rule-based score** used as a naive classifier,
  so the value-add of ML over a simple business formula is quantified, not assumed.

**Result:** ROC-AUC around 0.70–0.75 for the ML models, modestly beating the
rule-based benchmark (~0.70). This is a *realistic, credible* result for a noisy
operational forecasting problem — deliberately not "too good to be true."

## 6. Key Files to Know Before an Interview

| File | What it demonstrates |
|---|---|
| `src/generate_data.py` | Data modeling, understanding of ERP/SRM entity relationships, simulating realistic time-series shocks |
| `src/build_features.py` | Feature engineering, aggregation, rolling/trailing windows, leakage-safe target definition |
| `src/rule_based_score.py` | Translating business logic into transparent, defensible scoring — a core "analyst" skill |
| `src/train_model.py` | ML modeling, train/test methodology, evaluation metrics, benchmarking against a baseline |
| `dashboard/app.py` | Stakeholder communication, dashboard design, translating data into decisions |

## 7. Interview Preparation

### "Walk me through this project."
*"I built an end-to-end vendor risk monitoring system for a simulated electronics
manufacturer. It ingests purchase order, quality inspection, and disruption event
data, aggregates it into a vendor-quarter feature table, and produces two things: a
transparent weighted risk scorecard that procurement teams can actually understand
and defend to a vendor, and a machine learning model that predicts which vendors are
likely to become risky next quarter — so the team can act proactively instead of
reactively. I also built an interactive dashboard so a non-technical stakeholder could
actually use this."*

### "Why did you build both a rule-based score AND an ML model?"
*"In real vendor risk management, the rule-based score is what a procurement team
would actually put in front of a vendor in a business review — it's fully
explainable: 'your on-time delivery dropped, that's 25% of your score.' You can't do
that with a black-box model's output. But the rule-based score is backward-looking
and can't easily capture non-linear interactions. So I trained an ML model on the
same features to forecast forward-looking risk, and then benchmarked it directly
against the rule-based score to quantify whether the extra complexity was actually
worth it — which is the right way to justify using ML instead of just assuming it's
better."*

### "How did you avoid data leakage in the predictive model?"
*"Two things. First, the label — 'will this vendor breach next quarter' — is defined
using only data from the future quarter, and the model only ever sees features from
the current and prior quarters, never the quarter it's predicting. Second, I
evaluated with both a random train/test split and a time-based split, where I train
on earlier quarters and test only on later ones. That's the more rigorous approach
for a forecasting problem, because a random split can accidentally let the model
'see' patterns from a time period that, in production, wouldn't exist yet."*

### "Your ROC-AUC is only ~0.72. Is that a good result?"
*"For this kind of problem — noisy, real-world operational data with genuine
irreducible uncertainty — yes. If I'd gotten 0.98, I'd be suspicious of leakage
before I'd be impressed. 0.70–0.75 means the model is finding real, useful signal
without overfitting, and it modestly beats a well-designed business heuristic, which
is the realistic bar for this kind of forecasting problem in industry."*

### "What would you do differently in a production version?"
*"A few things: I'd want real financial data instead of a proxy score — likely from
a third-party risk data provider like Dun & Bradstreet or a service like Resilinc.
I'd add a feedback loop so the model retrains as new quarters land, and I'd want to
validate the risk score against actual downstream cost impact — if a 'high risk'
vendor rarely actually causes a production problem, the scoring weights need
recalibrating. I'd also want to run this against real historical incidents to
back-test whether the scorecard would have actually flagged past disruptions early
enough to matter."*

### "What's the hardest technical decision you made?"
*"Defining the prediction target without leakage. It's tempting to just predict
'is this vendor risky' using the same-quarter features that define risk — but that's
circular. I had to be disciplined about using only trailing/lagged features to
predict a *future* quarter's outcome, and I built in both a random and time-based
split specifically so I could validate that the model wasn't just memorizing
same-period correlations."*

### Resume bullet points (pick 2–3, tailor to the role)

- Built an end-to-end vendor risk analytics pipeline (Python, pandas, scikit-learn)
  processing 5,700+ purchase orders across 70 vendors to produce a transparent,
  weighted composite risk score and a predictive ML model forecasting next-quarter
  SLA breach risk (ROC-AUC 0.72, beating a rule-based benchmark).
- Designed a leakage-safe feature engineering pipeline with trailing/rolling
  performance signals, benchmarking a Random Forest classifier against a business
  rule-based scorecard to quantify the value of predictive modeling over static rules.
- Built an interactive Streamlit dashboard (Plotly) surfacing vendor risk heatmaps,
  scorecards, and an automated watchlist with recommended actions for procurement
  stakeholders.

## 8. Honest Limitations (know these — an interviewer respects this more than pretending it's perfect)

- All data is synthetic. It's realistically *structured* and *time-anchored*, but it
  doesn't reflect any real company's actual vendors.
- The financial "credit health score" is a proxy, not real credit bureau data.
- The rule-based weights (25/20/15/15/15/10) are a reasonable, defensible starting
  point but would need validation against real business outcomes (e.g., "does a high
  score actually predict cost-of-poor-quality?") in a production setting.
- ROC-AUC of ~0.72 means the model is useful for **prioritization** (who to review
  first), not a guarantee — it should support human judgment, not replace it.
