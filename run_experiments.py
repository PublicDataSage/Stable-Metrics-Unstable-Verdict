from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.datasets import load_dataset
from src.models import make_model
from src.metrics import predictive_metrics, fairness_metrics

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    return ap.parse_args()

def make_thresholds(cfg):
    t = cfg["operating_thresholds"]
    vals = np.arange(t["start"], t["stop"] + t["step"]/2, t["step"])
    return np.round(vals, 10)

def split_indices(y, seed, split_cfg):
    idx = np.arange(len(y))
    test_size = float(split_cfg["test"])
    val_size = float(split_cfg["validation"])

    train_val_idx, test_idx = train_test_split(
        idx, test_size=test_size, stratify=y, random_state=seed
    )
    y_tv = y.iloc[train_val_idx]
    rel_val = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=rel_val, stratify=y_tv, random_state=seed
    )
    return train_idx, val_idx, test_idx

def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    out_dir = Path(cfg.get("output_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    n_seeds = args.seeds or int(cfg["random_seeds"])
    datasets = [k for k,v in cfg["datasets"].items() if v.get("enabled", True)]
    if args.datasets:
        datasets = [d for d in datasets if d in set(args.datasets)]

    models = [k for k,v in cfg["models"].items() if v.get("enabled", True)]
    thresholds = make_thresholds(cfg)

    rows = []
    run_manifest = []

    for dname in datasets:
        print(f"[dataset] {dname}")
        bundle = load_dataset(dname, cfg["datasets"][dname])
        X, y = bundle.X, bundle.y

        for seed in range(n_seeds):
            train_idx, val_idx, test_idx = split_indices(y, seed, cfg["split"])
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

            for mname in models:
                print(f"  seed={seed:03d} model={mname}")
                model = make_model(mname, bundle.numeric_cols, bundle.categorical_cols, seed)
                model.fit(X_train, y_train)
                prob = model.predict_proba(X_test)[:, 1]

                for t in thresholds:
                    pred = (prob >= t).astype(int)
                    perf = predictive_metrics(y_test, prob, pred)

                    for attr, a_full in bundle.protected.items():
                        a_test = a_full.iloc[test_idx].to_numpy()
                        fm = fairness_metrics(
                            y_test.to_numpy(), pred, a_test,
                            min_group_n=cfg["analysis"].get("min_group_n", 20),
                            min_positive_group_n=cfg["analysis"].get("min_positive_group_n", 10),
                            min_negative_group_n=cfg["analysis"].get("min_negative_group_n", 10),
                        )

                        row = {
                            "dataset": dname,
                            "model": mname,
                            "seed": seed,
                            "protected_attribute": attr,
                            "operating_threshold": float(t),
                            "n_test": len(test_idx),
                            **perf,
                            **fm,
                        }
                        rows.append(row)

                run_manifest.append({
                    "dataset": dname,
                    "model": mname,
                    "seed": seed,
                    "n_train": len(train_idx),
                    "n_validation": len(val_idx),
                    "n_test": len(test_idx),
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "threshold_metrics.csv", index=False)
    pd.DataFrame(run_manifest).to_csv(out_dir / "run_manifest.csv", index=False)

    metadata = {
        "n_seeds": n_seeds,
        "datasets": datasets,
        "models": models,
        "operating_thresholds": list(map(float, thresholds)),
        "n_metric_rows": int(len(df)),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nSaved {len(df):,} rows to {out_dir / 'threshold_metrics.csv'}")

if __name__ == "__main__":
    main()
