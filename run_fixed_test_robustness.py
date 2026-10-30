#!/usr/bin/env python3
"""Fixed-test-set robustness experiment for fairness_audit_stability.

Run from the repository root. Uses existing src.datasets, src.models,
and src.metrics unchanged. Never writes into the original results directory.

Design:
  * One stratified 15% fixed test set per dataset, independent of run seed.
  * Per-seed training/validation split of the remaining 85%, yielding the
    same 70% training / 15% validation / 15% test fractions as the original.
  * Fixed test observations and protected attributes for every model/seed.
  * Each fitted model evaluated at all operating thresholds as before.
  * Atomic checkpoint per dataset/model/seed for resumable runs.
  * Output schema compatible with analyze_experiments.py.

Causal scope: Holds evaluation-cohort composition fixed. Variation in the
models may still combine development-data resampling and estimator RNG.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.datasets import load_dataset
from src.metrics import fairness_metrics, predictive_metrics
from src.models import make_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="results/fixed_test_robustness")
    parser.add_argument("--seeds", type=int, default=None,
                        help="Number of model seeds starting from zero")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Dataset names from config.yaml")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model names from config.yaml")
    parser.add_argument("--fixed-test-seed", type=int, default=2027,
                        help="Seed used only for the shared test partition")
    return parser.parse_args()


def atomic_csv(df: pd.DataFrame, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(target)


def atomic_json(obj: dict, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(target)


def dataset_fingerprint(bundle) -> str:
    """Detect changes in row ordering or dataset values between reruns."""
    h = hashlib.sha256()
    for obj in (bundle.X, bundle.y, *bundle.protected.values()):
        hashes = pd.util.hash_pandas_object(obj, index=False)
        h.update(hashes.to_numpy(dtype=np.uint64).tobytes())
    return h.hexdigest()


def thresholds_from_config(cfg):
    tcfg = cfg["operating_thresholds"]
    step = float(tcfg["step"])
    if step <= 0:
        raise ValueError("operating_thresholds.step must be positive")
    return np.round(np.arange(float(tcfg["start"]),
                              float(tcfg["stop"]) + step / 2, step), 10)


def ensure_protocol(out: Path, cfg: dict, fixed_test_seed: int):
    """Reject accidental resume with changed protocol/settings."""
    effective = {
        "fixed_test_seed": int(fixed_test_seed),
        "split": cfg["split"],
        "datasets": cfg["datasets"],
        "models": cfg["models"],
        "operating_thresholds": cfg["operating_thresholds"],
        "analysis_support": {
            key: cfg["analysis"].get(key, default)
            for key, default in (
                ("min_group_n", 20),
                ("min_positive_group_n", 10),
                ("min_negative_group_n", 10),
            )
        },
    }
    protocol_path = out / "run_protocol.json"
    if protocol_path.exists():
        prior = json.loads(protocol_path.read_text(encoding="utf-8"))
        if prior != effective:
            raise RuntimeError(
                "Existing output uses a different protocol. Choose a new --out "
                "folder to prevent mixing results."
            )
    else:
        atomic_json(effective, protocol_path)


def fixed_indices(bundle, name: str, cfg: dict, fixed_test_seed: int, out: Path):
    y = bundle.y
    idx = np.arange(len(y), dtype=np.int64)
    trainval_idx, test_idx = train_test_split(
        idx,
        test_size=float(cfg["split"]["test"]),
        stratify=y.iloc[idx],
        random_state=fixed_test_seed,
    )
    test_idx = np.sort(np.asarray(test_idx, dtype=np.int64))
    trainval_idx = np.sort(np.asarray(trainval_idx, dtype=np.int64))

    saved = out / f"fixed_test_indices_{name}.csv"
    saved_meta = out / f"fixed_test_indices_{name}.json"
    fp = dataset_fingerprint(bundle)
    if saved.exists() != saved_meta.exists():
        raise RuntimeError(f"Incomplete fixed-test index metadata for {name}")
    if saved.exists():
        prev = pd.read_csv(saved)["row_position"].to_numpy(dtype=np.int64)
        meta = json.loads(saved_meta.read_text(encoding="utf-8"))
        if not np.array_equal(prev, test_idx) or meta["dataset_sha256"] != fp:
            raise RuntimeError(
                f"Fixed test partition or dataset content changed for {name}. "
                "Use a fresh --out directory."
            )
    else:
        atomic_csv(pd.DataFrame({"row_position": test_idx}), saved)
        atomic_json({
            "dataset": name,
            "dataset_sha256": fp,
            "n_rows": int(len(idx)),
            "n_test": int(len(test_idx)),
            "fixed_test_seed": int(fixed_test_seed),
            "label_test_counts": {
                str(k): int(v) for k, v in
                y.iloc[test_idx].value_counts(dropna=False).items()
            },
        }, saved_meta)
    return trainval_idx, test_idx


def trainval_split(bundle, trainval_idx, seed: int, cfg: dict):
    # Among the 85% development set, 15/85 goes to validation.
    relative_val = float(cfg["split"]["validation"]) / (
        float(cfg["split"]["train"]) + float(cfg["split"]["validation"]))
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=relative_val,
        stratify=bundle.y.iloc[trainval_idx],
        random_state=seed,
    )
    return np.asarray(train_idx), np.asarray(val_idx)


def run_one(bundle, name, model_name, seed, train_idx, val_idx, test_idx,
            thresholds, cfg, fixed_test_seed):
    if np.intersect1d(train_idx, test_idx).size:
        raise AssertionError("Data leakage: training overlaps the test set")
    if np.intersect1d(val_idx, test_idx).size:
        raise AssertionError("Data leakage: validation overlaps the test set")
    if np.intersect1d(train_idx, val_idx).size:
        raise AssertionError("Data leakage: training overlaps validation")

    X_train, y_train = bundle.X.iloc[train_idx], bundle.y.iloc[train_idx]
    X_test, y_test = bundle.X.iloc[test_idx], bundle.y.iloc[test_idx]
    y_test_np = y_test.to_numpy()
    protected_test = {
        attr: values.iloc[test_idx].to_numpy()
        for attr, values in bundle.protected.items()
    }

    model = make_model(
        model_name, bundle.numeric_cols, bundle.categorical_cols, seed
    )
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]

    support = cfg.get("analysis", {})
    rows = []
    for t in thresholds:
        prediction = (prob >= t).astype(int)
        perf = predictive_metrics(y_test_np, prob, prediction)
        for attr, groups in protected_test.items():
            fm = fairness_metrics(
                y_test_np, prediction, groups,
                min_group_n=support.get("min_group_n", 20),
                min_positive_group_n=support.get("min_positive_group_n", 10),
                min_negative_group_n=support.get("min_negative_group_n", 10),
            )
            rows.append({
                "dataset": name,
                "model": model_name,
                "seed": int(seed),
                "protected_attribute": attr,
                "operating_threshold": float(t),
                "n_test": int(len(test_idx)),
                **perf,
                **fm,
            })

    manifest = {
        "dataset": name,
        "model": model_name,
        "seed": int(seed),
        "n_train": int(len(train_idx)),
        "n_validation": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "fixed_test_seed": int(fixed_test_seed),
    }
    return pd.DataFrame(rows), manifest


def collect_results(out, chosen_datasets, chosen_models, n_seeds,
                    thresholds, attributes):
    data_parts, manifest_rows = [], []
    for name in chosen_datasets:
        for model in chosen_models:
            for seed in range(n_seeds):
                path = out / "checkpoints" / name / model / f"seed_{seed:03d}.csv"
                mpath = path.with_suffix(".json")
                if not path.exists() or not mpath.exists():
                    raise RuntimeError(f"Missing complete checkpoint: {path}")
                part = pd.read_csv(path)
                expected = len(thresholds) * len(attributes[name])
                if len(part) != expected:
                    raise RuntimeError(f"Checkpoint row count mismatch: {path}")
                data_parts.append(part)
                manifest_rows.append(json.loads(mpath.read_text(encoding="utf-8")))
    data = pd.concat(data_parts, ignore_index=True)
    manifest = pd.DataFrame(manifest_rows)
    atomic_csv(data, out / "threshold_metrics.csv")
    atomic_csv(manifest, out / "run_manifest.csv")
    return data, manifest


def main():
    args = parse_args()
    if args.seeds is not None and args.seeds <= 0:
        raise ValueError("--seeds must be positive")
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    split = cfg["split"]
    if not np.isclose(sum(map(float, split.values())), 1.0):
        raise ValueError("Split fractions must sum to 1")
    if any(float(v) <= 0 for v in split.values()):
        raise ValueError("All train/validation/test fractions must be positive")

    out = Path(args.out).resolve()
    original_out = Path(cfg.get("output_dir", "results")).resolve()
    if out == original_out or out == original_out.parent:
        raise ValueError("Refusing to overwrite the original results directory")
    out.mkdir(parents=True, exist_ok=True)
    ensure_protocol(out, cfg, args.fixed_test_seed)

    enabled_datasets = [
        key for key, val in cfg["datasets"].items() if val.get("enabled", True)
    ]
    enabled_models = [
        key for key, val in cfg["models"].items() if val.get("enabled", True)
    ]
    datasets = args.datasets if args.datasets else enabled_datasets
    models = args.models if args.models else enabled_models
    if not datasets or not models:
        raise ValueError("No datasets or models selected")
    if set(datasets) - set(enabled_datasets):
        raise ValueError(f"Unknown/disabled datasets: {set(datasets)-set(enabled_datasets)}")
    if set(models) - set(enabled_models):
        raise ValueError(f"Unknown/disabled models: {set(models)-set(enabled_models)}")

    n_seeds = args.seeds if args.seeds is not None else int(cfg["random_seeds"])
    thresholds = thresholds_from_config(cfg)
    attributes = {}

    for name in datasets:
        print(f"\n[dataset] {name}", flush=True)
        bundle = load_dataset(name, cfg["datasets"][name])
        dev_idx, test_idx = fixed_indices(
            bundle, name, cfg, args.fixed_test_seed, out
        )
        attributes[name] = list(bundle.protected.keys())
        print(f"  Fixed test: {len(test_idx)} rows; development: {len(dev_idx)} rows")
        for seed in range(n_seeds):
            train_idx, val_idx = trainval_split(bundle, dev_idx, seed, cfg)
            for model in models:
                checkpoint = out / "checkpoints" / name / model / f"seed_{seed:03d}.csv"
                checkpoint_meta = checkpoint.with_suffix(".json")
                if checkpoint.exists() and checkpoint_meta.exists():
                    print(f"  [skip] seed={seed:03d} model={model}", flush=True)
                    continue
                print(f"  [fit]  seed={seed:03d} model={model}", flush=True)
                data, manifest = run_one(
                    bundle, name, model, seed, train_idx, val_idx,
                    test_idx, thresholds, cfg, args.fixed_test_seed,
                )
                atomic_csv(data, checkpoint)
                atomic_json(manifest, checkpoint_meta)

    data, manifest = collect_results(
        out, datasets, models, n_seeds, thresholds, attributes
    )
    analysis_cfg = dict(cfg)
    analysis_cfg["output_dir"] = str(out)
    (out / "config_fixed_test.yaml").write_text(
        yaml.safe_dump(analysis_cfg, sort_keys=False), encoding="utf-8"
    )
    atomic_json({
        "design": "fixed_test; seeded development split and model fitting",
        "n_seeds": n_seeds,
        "fixed_test_seed": args.fixed_test_seed,
        "datasets": datasets,
        "models": models,
        "operating_thresholds": list(map(float, thresholds)),
        "n_metric_rows": int(len(data)),
        "n_fitted_models": int(len(manifest)),
        "output_dir": str(out),
    }, out / "metadata.json")
    print(f"\n[done] {len(manifest):,} fitted models")
    print(f"[done] {len(data):,} metric rows -> {out / 'threshold_metrics.csv'}")
    print("\nAnalyze with:")
    print(f'  python analyze_experiments.py --config "{out / "config_fixed_test.yaml"}"')


if __name__ == "__main__":
    main()
