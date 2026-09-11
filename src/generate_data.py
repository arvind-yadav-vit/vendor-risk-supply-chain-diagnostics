"""
generate_data.py
=================
Generates a synthetic but REALISTIC dataset simulating an electronics/tech
hardware manufacturer's supply chain: vendors, purchase orders/shipments,
quality inspections, financial-health signals, and disruption events.

WHY SYNTHETIC DATA (and why it's still a legitimate portfolio project):
Real vendor performance data is confidential to every company that has it,
so no public "supply chain risk" dataset exists that looks like what an
actual SRM/procurement team uses. Instead of grabbing a generic Kaggle CSV,
we simulate the data generation PROCESS the way it would actually occur in
an ERP/SRM system (SAP Ariba, Coupa, GEP style), with realistic entities,
relationships, seasonality, and injected disruption events tied to real
supply-chain risk patterns from 2022-2026 (chip shortage tail, Red Sea
shipping disruption, tariff escalations, regional plant shutdowns).

Run:
    python src/generate_data.py

Outputs (in /data):
    vendors.csv
    purchase_orders.csv
    quality_inspections.csv
    disruption_events.csv
    vendor_financial_signals.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# ----------------------------------------------------------------------
# 0. REPRODUCIBILITY
# ----------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

OUT_DIR = "data"

# ----------------------------------------------------------------------
# 1. REFERENCE DATA — the "master data" of the supply chain
# ----------------------------------------------------------------------

COMPONENT_CATEGORIES = {
    "Semiconductors":      {"base_lead_time": 95, "lead_time_std": 25, "unit_cost_range": (2, 180)},
    "PCB / PCBA":          {"base_lead_time": 45, "lead_time_std": 12, "unit_cost_range": (5, 90)},
    "Passive Components":  {"base_lead_time": 30, "lead_time_std": 8,  "unit_cost_range": (0.02, 3)},
    "Connectors":          {"base_lead_time": 35, "lead_time_std": 10, "unit_cost_range": (0.5, 15)},
    "Displays":            {"base_lead_time": 60, "lead_time_std": 15, "unit_cost_range": (8, 220)},
    "Batteries":           {"base_lead_time": 50, "lead_time_std": 14, "unit_cost_range": (3, 60)},
    "Enclosures / Metal":  {"base_lead_time": 40, "lead_time_std": 10, "unit_cost_range": (1, 45)},
    "Packaging":           {"base_lead_time": 20, "lead_time_std": 6,  "unit_cost_range": (0.1, 5)},
}

# Countries weighted toward real electronics manufacturing hubs
COUNTRY_REGION = {
    "Taiwan": "APAC", "China": "APAC", "South Korea": "APAC", "Vietnam": "APAC",
    "Malaysia": "APAC", "Japan": "APAC", "India": "APAC", "Thailand": "APAC",
    "Germany": "EMEA", "Poland": "EMEA", "Czech Republic": "EMEA", "Morocco": "EMEA",
    "Mexico": "AMER", "United States": "AMER", "Brazil": "AMER",
}
COUNTRIES = list(COUNTRY_REGION.keys())
# Rough concentration mirroring real electronics sourcing (APAC-heavy)
COUNTRY_WEIGHTS = [0.16, 0.14, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04,
                   0.06, 0.04, 0.03, 0.02, 0.06, 0.05, 0.04]
COUNTRY_WEIGHTS = np.array(COUNTRY_WEIGHTS) / np.sum(COUNTRY_WEIGHTS)

PLANTS = ["Plant-Austin-US", "Plant-Guadalajara-MX", "Plant-Penang-MY", "Plant-Brno-CZ"]

CERTS = ["ISO9001", "ISO14001", "IATF16949", "RoHS", "REACH"]

N_VENDORS = 70

# ----------------------------------------------------------------------
# 2. VENDOR MASTER TABLE
# ----------------------------------------------------------------------
def generate_vendors():
    rows = []
    for i in range(1, N_VENDORS + 1):
        vendor_id = f"V{i:04d}"
        category = random.choice(list(COMPONENT_CATEGORIES.keys()))
        country = np.random.choice(COUNTRIES, p=COUNTRY_WEIGHTS)
        region = COUNTRY_REGION[country]

        # Tier: Tier-1 = strategic/high-spend, fewer vendors; Tier-3 = long tail
        tier = np.random.choice(["Tier 1", "Tier 2", "Tier 3"], p=[0.15, 0.35, 0.50])

        onboarding_date = datetime(2018, 1, 1) + timedelta(days=int(np.random.uniform(0, 365 * 5)))

        # Annual spend correlates loosely with tier
        spend_base = {"Tier 1": (2_000_000, 15_000_000),
                      "Tier 2": (400_000, 2_000_000),
                      "Tier 3": (20_000, 400_000)}[tier]
        annual_spend = round(np.random.uniform(*spend_base), -2)

        n_certs = np.random.choice([0, 1, 2, 3, 4], p=[0.10, 0.20, 0.30, 0.25, 0.15])
        certifications = ", ".join(sorted(random.sample(CERTS, n_certs))) if n_certs else "None"

        # Single-source flag: vendor is the ONLY supplier for a critical part.
        # More common (and riskier) for semiconductors / displays, and Tier 1.
        single_source_prob = 0.35 if category in ["Semiconductors", "Displays"] else 0.12
        single_source_flag = np.random.rand() < single_source_prob

        # Underlying "true" vendor quality archetype — this is a LATENT variable
        # that drives performance below. It is intentionally NOT put in the
        # final dataset directly (models must infer risk from behavior, not
        # from a label we hand them — this avoids target leakage).
        archetype = np.random.choice(
            ["excellent", "solid", "average", "shaky", "high_risk"],
            p=[0.12, 0.28, 0.32, 0.18, 0.10]
        )

        rows.append({
            "vendor_id": vendor_id,
            "vendor_name": f"{category.split(' ')[0][:4].upper()}-{country[:3].upper()}-{i}",
            "category": category,
            "country": country,
            "region": region,
            "tier": tier,
            "onboarding_date": onboarding_date.date(),
            "annual_spend_usd": annual_spend,
            "certifications": certifications,
            "single_source_flag": single_source_flag,
            "_archetype": archetype,  # kept with underscore, used only for generation
        })
    return pd.DataFrame(rows)


vendors_df = generate_vendors()

# Archetype -> base performance parameters (this is the "ground truth"
# generative model behind the numbers you'll see in the CSVs)
ARCHETYPE_PARAMS = {
    "excellent":  {"otd_mean": 0.97, "defect_mean": 0.003, "cost_var_std": 0.02, "disruption_rate": 0.03},
    "solid":      {"otd_mean": 0.93, "defect_mean": 0.008, "cost_var_std": 0.04, "disruption_rate": 0.06},
    "average":    {"otd_mean": 0.87, "defect_mean": 0.018, "cost_var_std": 0.07, "disruption_rate": 0.10},
    "shaky":      {"otd_mean": 0.76, "defect_mean": 0.035, "cost_var_std": 0.12, "disruption_rate": 0.18},
    "high_risk":  {"otd_mean": 0.62, "defect_mean": 0.065, "cost_var_std": 0.20, "disruption_rate": 0.30},
}

# ----------------------------------------------------------------------
# 3. MACRO / MARKET EVENTS — realistic time-anchored shocks
#    (Inspired by real 2022-2026 supply chain events; dates approximate)
# ----------------------------------------------------------------------
MACRO_EVENTS = [
    # (start, end, affected_regions, affected_categories, severity_mult, label)
    (datetime(2022, 1, 1), datetime(2022, 12, 31), ["APAC"], ["Semiconductors"], 1.8, "Global chip shortage tail"),
    (datetime(2022, 3, 1), datetime(2022, 6, 30), ["APAC"], None, 1.6, "China regional lockdowns"),
    (datetime(2023, 11, 1), datetime(2024, 5, 31), ["APAC", "EMEA"], None, 1.5, "Red Sea shipping route disruption"),
    (datetime(2024, 4, 1), datetime(2024, 7, 31), ["APAC"], ["Semiconductors", "Displays"], 1.4, "Taiwan seismic + grid strain"),
    (datetime(2025, 2, 1), datetime(2025, 12, 31), None, None, 1.3, "Escalating tariff environment"),
    (datetime(2026, 1, 1), datetime(2026, 6, 30), ["AMER"], ["Enclosures / Metal", "Packaging"], 1.2, "Steel/aluminum tariff pass-through"),
]

def macro_multiplier(date, region, category):
    mult = 1.0
    for start, end, regions, cats, sev, _ in MACRO_EVENTS:
        if start <= date <= end:
            region_hit = regions is None or region in regions
            cat_hit = cats is None or category in cats
            if region_hit and cat_hit:
                mult *= sev
    return mult

# ----------------------------------------------------------------------
# 4. PURCHASE ORDERS / SHIPMENTS
# ----------------------------------------------------------------------
def generate_purchase_orders(vendors_df, start=datetime(2022, 1, 1), end=datetime(2026, 6, 30)):
    po_rows = []
    po_counter = 1
    total_days = (end - start).days

    for _, v in vendors_df.iterrows():
        params = ARCHETYPE_PARAMS[v["_archetype"]]
        cat_info = COMPONENT_CATEGORIES[v["category"]]

        # Higher tier / higher spend vendors get more PO volume
        n_orders = {"Tier 1": np.random.randint(180, 260),
                    "Tier 2": np.random.randint(60, 140),
                    "Tier 3": np.random.randint(15, 50)}[v["tier"]]

        order_days = np.sort(np.random.choice(range(total_days), size=n_orders, replace=False))

        for d in order_days:
            order_date = start + timedelta(days=int(d))
            plant = random.choice(PLANTS)

            base_lead = cat_info["base_lead_time"]
            lead_std = cat_info["lead_time_std"]
            macro_mult = macro_multiplier(order_date, v["region"], v["category"])

            # Promised lead time (what vendor commits to)
            promised_lead = max(5, int(np.random.normal(base_lead, lead_std * 0.4)))
            promised_delivery = order_date + timedelta(days=promised_lead)

            # Actual delivery: driven by archetype OTD reliability + macro shocks
            on_time_prob = params["otd_mean"] / macro_mult
            on_time_prob = min(max(on_time_prob, 0.05), 0.99)
            is_on_time = np.random.rand() < on_time_prob

            if is_on_time:
                delay_days = max(0, int(np.random.normal(0, 1.5)))
            else:
                # Delay severity scales with macro disruption
                delay_days = int(np.random.gamma(shape=2.0, scale=6 * macro_mult))

            actual_delivery = promised_delivery + timedelta(days=delay_days)

            qty_ordered = int(np.random.lognormal(mean=8.5, sigma=1.2))
            qty_ordered = max(50, min(qty_ordered, 200_000))

            # Short-shipments more likely for high-risk vendors / disrupted periods
            short_ship_prob = (1 - params["otd_mean"]) * 0.5 * macro_mult
            if np.random.rand() < min(short_ship_prob, 0.4):
                qty_received = int(qty_ordered * np.random.uniform(0.7, 0.98))
            else:
                qty_received = qty_ordered

            unit_cost_lo, unit_cost_hi = cat_info["unit_cost_range"]
            quoted_cost = round(np.random.uniform(unit_cost_lo, unit_cost_hi), 3)

            # Actual cost variance — inflation/tariff pass-through + vendor cost discipline
            cost_var_pct = np.random.normal(0, params["cost_var_std"]) * macro_mult
            actual_cost = round(max(0.001, quoted_cost * (1 + cost_var_pct)), 3)

            incoterm = random.choice(["FOB", "CIF", "EXW", "DDP"])
            transport_mode = random.choice(["Ocean", "Air", "Ocean", "Ocean", "Truck"])
            expedited = transport_mode == "Air" and delay_days == 0 and np.random.rand() < 0.3

            po_rows.append({
                "po_id": f"PO{po_counter:07d}",
                "vendor_id": v["vendor_id"],
                "component_category": v["category"],
                "receiving_plant": plant,
                "order_date": order_date.date(),
                "promised_delivery_date": promised_delivery.date(),
                "actual_delivery_date": actual_delivery.date(),
                "delay_days": delay_days,
                "quantity_ordered": qty_ordered,
                "quantity_received": qty_received,
                "quoted_unit_cost_usd": quoted_cost,
                "actual_unit_cost_usd": actual_cost,
                "incoterm": incoterm,
                "transport_mode": transport_mode,
                "expedited_flag": expedited,
            })
            po_counter += 1

    return pd.DataFrame(po_rows)


purchase_orders_df = generate_purchase_orders(vendors_df)

# ----------------------------------------------------------------------
# 5. QUALITY INSPECTIONS (linked to a sample of POs — not every PO is inspected)
# ----------------------------------------------------------------------
def generate_quality_inspections(purchase_orders_df, vendors_df):
    vendor_lookup = vendors_df.set_index("vendor_id")
    rows = []
    inspection_counter = 1

    # Inspect a sample of POs (incoming quality control doesn't check 100%)
    sampled = purchase_orders_df.sample(frac=0.65, random_state=SEED)

    NC_TYPES = ["Dimensional deviation", "Functional failure", "Cosmetic defect",
                "Labeling/documentation error", "Contamination", "Packaging damage"]

    for _, po in sampled.iterrows():
        v = vendor_lookup.loc[po["vendor_id"]]
        params = ARCHETYPE_PARAMS[v["_archetype"]]
        macro_mult = macro_multiplier(
            datetime.combine(po["actual_delivery_date"], datetime.min.time()),
            v["region"], v["category"]
        )

        defect_rate = max(0, np.random.normal(params["defect_mean"], params["defect_mean"] * 0.6) * (1 + 0.3 * (macro_mult - 1)))
        defect_rate = min(defect_rate, 0.5)

        lot_rejected = defect_rate > np.random.uniform(0.03, 0.08)

        corrective_action_days = None
        nc_type = None
        if lot_rejected:
            nc_type = random.choice(NC_TYPES)
            # Weaker vendors take longer to resolve corrective actions (CAPA)
            corrective_action_days = int(np.random.gamma(shape=2, scale=6 / max(params["otd_mean"], 0.3)))

        rows.append({
            "inspection_id": f"QI{inspection_counter:07d}",
            "po_id": po["po_id"],
            "vendor_id": po["vendor_id"],
            "inspection_date": po["actual_delivery_date"],
            "defect_rate": round(defect_rate, 4),
            "lot_rejected_flag": lot_rejected,
            "non_conformance_type": nc_type,
            "corrective_action_days": corrective_action_days,
        })
        inspection_counter += 1

    return pd.DataFrame(rows)


quality_inspections_df = generate_quality_inspections(purchase_orders_df, vendors_df)

# ----------------------------------------------------------------------
# 6. DISRUPTION / RISK EVENTS (discrete, dated incidents per vendor)
# ----------------------------------------------------------------------
def generate_disruption_events(vendors_df, start=datetime(2022, 1, 1), end=datetime(2026, 6, 30)):
    EVENT_TYPES = ["Geopolitical / trade restriction", "Natural disaster", "Financial distress",
                   "Labor strike", "Cyber incident", "Regulatory non-compliance", "Plant fire/safety incident"]
    rows = []
    event_counter = 1
    total_days = (end - start).days

    for _, v in vendors_df.iterrows():
        params = ARCHETYPE_PARAMS[v["_archetype"]]
        # Expected number of events over the whole window scales with archetype disruption rate
        expected_events = params["disruption_rate"] * (total_days / 365) * 1.3
        n_events = np.random.poisson(expected_events)

        for _ in range(n_events):
            event_date = start + timedelta(days=int(np.random.uniform(0, total_days)))
            macro_mult = macro_multiplier(event_date, v["region"], v["category"])

            severity = np.random.choice(["Low", "Medium", "High", "Critical"],
                                         p=np.array([0.40, 0.35, 0.20, 0.05]) if macro_mult < 1.3
                                         else np.array([0.20, 0.35, 0.32, 0.13]))

            resolution_days = {"Low": np.random.randint(1, 7),
                                "Medium": np.random.randint(5, 21),
                                "High": np.random.randint(14, 60),
                                "Critical": np.random.randint(30, 120)}[severity]

            rows.append({
                "event_id": f"EV{event_counter:06d}",
                "vendor_id": v["vendor_id"],
                "event_date": event_date.date(),
                "event_type": random.choice(EVENT_TYPES),
                "severity": severity,
                "region_affected": v["region"],
                "resolution_days": resolution_days,
            })
            event_counter += 1

    return pd.DataFrame(rows)


disruption_events_df = generate_disruption_events(vendors_df)

# ----------------------------------------------------------------------
# 7. QUARTERLY FINANCIAL / MARKET SIGNALS (external risk-data-provider style)
# ----------------------------------------------------------------------
def generate_financial_signals(vendors_df, start=datetime(2022, 1, 1), end=datetime(2026, 6, 30)):
    rows = []
    quarters = pd.period_range(start=start, end=end, freq="Q")

    for _, v in vendors_df.iterrows():
        params = ARCHETYPE_PARAMS[v["_archetype"]]
        # Base credit proxy: excellent archetypes run high & stable; high_risk run low & volatile
        base_score = {"excellent": 82, "solid": 73, "average": 62, "shaky": 48, "high_risk": 34}[v["_archetype"]]
        dpo_base = {"excellent": 45, "solid": 50, "average": 58, "shaky": 70, "high_risk": 85}[v["_archetype"]]

        score = base_score
        for q in quarters:
            q_date = q.to_timestamp()
            macro_mult = macro_multiplier(q_date, v["region"], v["category"])

            # Random walk with mean reversion + macro drag
            drift = np.random.normal(0, 3) - (macro_mult - 1) * 8
            score = float(np.clip(score * 0.9 + (base_score + drift) * 0.1, 5, 99))

            dpo = max(15, dpo_base + np.random.normal(0, 6) + (macro_mult - 1) * 15)
            price_volatility_index = round(params["cost_var_std"] * 100 * macro_mult + np.random.normal(0, 2), 2)

            rows.append({
                "vendor_id": v["vendor_id"],
                "quarter": str(q),
                "credit_risk_score_proxy": round(score, 1),  # 0-100, HIGHER = healthier (like a credit score)
                "days_payable_outstanding": round(dpo, 1),
                "price_volatility_index": max(0, price_volatility_index),
            })

    return pd.DataFrame(rows)


financial_signals_df = generate_financial_signals(vendors_df)

# ----------------------------------------------------------------------
# 8. SAVE — drop the "_archetype" ground-truth column from the public
#    vendors table (it's an internal generator variable, not something a
#    real dataset would hand you — this keeps the ML problem honest).
# ----------------------------------------------------------------------
vendors_public = vendors_df.drop(columns=["_archetype"])

vendors_public.to_csv(f"{OUT_DIR}/vendors.csv", index=False)
purchase_orders_df.to_csv(f"{OUT_DIR}/purchase_orders.csv", index=False)
quality_inspections_df.to_csv(f"{OUT_DIR}/quality_inspections.csv", index=False)
disruption_events_df.to_csv(f"{OUT_DIR}/disruption_events.csv", index=False)
financial_signals_df.to_csv(f"{OUT_DIR}/vendor_financial_signals.csv", index=False)

# Keep archetype mapping SEPARATELY for our own validation use only
vendors_df[["vendor_id", "_archetype"]].to_csv(f"{OUT_DIR}/_ground_truth_archetype.csv", index=False)

print("Data generation complete.")
print(f"  vendors.csv                 : {len(vendors_public):>7,} rows")
print(f"  purchase_orders.csv         : {len(purchase_orders_df):>7,} rows")
print(f"  quality_inspections.csv     : {len(quality_inspections_df):>7,} rows")
print(f"  disruption_events.csv       : {len(disruption_events_df):>7,} rows")
print(f"  vendor_financial_signals.csv: {len(financial_signals_df):>7,} rows")
