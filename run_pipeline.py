"""
run_pipeline.py
================
Runs the full project pipeline end-to-end, in order:
    1. Generate synthetic supply chain data
    2. Build vendor-quarter feature table
    3. Compute rule-based composite risk scores
    4. Train & evaluate ML risk models

Run once from the project root:
    python run_pipeline.py

Then launch the dashboard separately:
    streamlit run dashboard/app.py
"""
import subprocess
import sys

STEPS = [
    ("Generating synthetic supply chain data...", "src/generate_data.py"),
    ("Building vendor-quarter feature table...", "src/build_features.py"),
    ("Computing rule-based composite risk scores...", "src/rule_based_score.py"),
    ("Training & evaluating ML risk models...", "src/train_model.py"),
]

for message, script in STEPS:
    print("\n" + "=" * 70)
    print(message)
    print("=" * 70)
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"\nStep failed: {script}")
        sys.exit(1)

print("\n" + "=" * 70)
print("Pipeline complete. Launch the dashboard with:")
print("    streamlit run dashboard/app.py")
print("=" * 70)
