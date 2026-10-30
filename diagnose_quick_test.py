from pathlib import Path
import pandas as pd

path=Path("results/threshold_metrics.csv")
df=pd.read_csv(path)
print("\n=== STRUCTURE ===")
print("Rows:",len(df))
print("Datasets:",sorted(df.dataset.unique()))
print("Models:",sorted(df.model.unique()))
print("Seeds:",sorted(df.seed.unique()))
print("Thresholds:",sorted(df.operating_threshold.unique()))
print("Protected attributes:",sorted(df.protected_attribute.unique()))
print("\n=== GROUP SUPPORT ===")
print(df.groupby(["dataset","protected_attribute"])[["n_min_group","n_min_positive_group","n_min_negative_group"]].agg(["min","median","max"]))
print("\n=== VALIDITY COUNTS ===")
for c in ["dpd_analysis_valid","dpr_analysis_valid","eod_analysis_valid","equalized_odds_analysis_valid"]:
    print(c,df[c].value_counts(dropna=False).to_dict())
print("\n=== UNDEFINED FAIRNESS COUNTS ===")
for c in ["dpd_violation","dpr_violation","eod_violation","equalized_odds_violation"]:
    print(c,int(df[c].isna().sum()))
