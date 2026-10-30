#!/usr/bin/env python3
"""Statistical validation for fairness_audit_stability.

This script READS existing experiment outputs; it never retrains models and
never overwrites original/fixed-test experiment outputs.

Expected project structure:
    results/experiment1_retraining_reproducibility.csv
    results/experiment3_audit_stability_surface.csv
    results/fixed_test_robustness/threshold_metrics.csv
    results/fixed_test_robustness/experiment1_retraining_reproducibility.csv
    results/fixed_test_robustness/experiment3_audit_stability_surface.csv
    results/fixed_test_robustness/experiment4_predictive_vs_audit_stability.csv

Outputs are written exclusively to --out. 95% Wilson intervals are pointwise
under independent-seed Bernoulli sampling; repeated tolerance values are NOT
independent. Bootstrap intervals are descriptive across dataset/model blocks,
not confidence intervals generalizable to all fairness applications.
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

FAMILY = ["dataset", "model", "protected_attribute", "fairness_metric"]
KEY = FAMILY + ["operating_threshold", "audit_tolerance"]
E1KEY = FAMILY + ["audit_tolerance"]
METRICS = ("dpd_violation", "dpr_violation", "eod_violation", "equalized_odds_violation")
ESTIMATORS = {
    "ols_contextual": "experiment4_clustered_ols_contextual.txt",
    "ols_adjusted": "experiment4_clustered_ols_mechanism_adjusted.txt",
    "gee_fractional_exchangeable": "experiment4_fractional_gee.txt",
    "gee_binary_exchangeable": "experiment4_binary_unstable_gee.txt",
}


def arguments():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original-dir", type=Path, default=Path("results"),
                    help="Existing original results directory (optional if absent)")
    ap.add_argument("--fixed-dir", type=Path, default=Path("results/fixed_test_robustness"),
                    help="Existing fixed-test-set results directory")
    ap.add_argument("--out", type=Path, default=Path("results/statistical_validation"))
    ap.add_argument("--primary-t", type=float, default=0.5)
    ap.add_argument("--epsilon", type=float, default=0.10)
    ap.add_argument("--bootstrap", type=int, default=1000,
                    help="Cluster bootstrap repetitions for descriptive matched comparison")
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--skip-regressions", action="store_true")
    ap.add_argument("--require-original", action="store_true",
                    help="Fail if original surface is missing rather than skip comparison")
    return ap.parse_args()


def csv_read(path, needed, *, keys=None):
    if not path.is_file():
        raise FileNotFoundError(f"Required input missing: {path}")
    df = pd.read_csv(path, low_memory=False)
    absent = set(needed) - set(df.columns)
    if absent:
        raise ValueError(f"{path}: missing columns {sorted(absent)}")
    for c in ("operating_threshold", "audit_tolerance"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="raise").round(9)
    if keys is not None:
        bad = df.duplicated(keys, keep=False)
        if bad.any():
            raise ValueError(f"{path}: {bad.sum()} rows have duplicate keys {keys}")
    return df


def save_csv(df, path):
    df.to_csv(path, index=False, float_format="%.10g")
    print(f"  Saved: {path} ({len(df):,} rows)")


def check(flag, message, log):
    log.append({"status": "PASS" if flag else "FAIL", "check": message})
    print(f"  [{'PASS' if flag else 'FAIL'}] {message}")


def check_surface(df, label, log):
    p = df["pass_probability"].to_numpy(float)
    vi = df["verdict_instability"].to_numpy(float)
    n = df["n_retrainings"].to_numpy(float)
    check(np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all(), f"{label}: 0<=p<=1", log)
    check(np.isfinite(vi).all() and np.allclose(vi, 2*p*(1-p), atol=1e-9),
          f"{label}: VI=2p(1-p)", log)
    check(np.isfinite(n).all() and (n >= 2).all() and np.allclose(n, np.round(n)),
          f"{label}: valid retraining counts >=2", log)
    check(np.allclose(n*p, np.round(n*p), atol=1e-5),
          f"{label}: reported p corresponds to integer pass counts", log)
    if "region" in df:
        label_map = np.where(p >= 0.9, "stable_within_tolerance",
                    np.where(p <= 0.1, "stable_outside_tolerance", "unstable"))
        # Original output labels assume epsilon=0.10; check independently of CLI epsilon.
        check(np.array_equal(df["region"].to_numpy(str), label_map),
              f"{label}: region labels consistent with epsilon=0.10", log)


def validate_fixed_inputs(fixed, t0, out, log):
    print("\n[1] Fixed-test design and summary integrity")
    metrics = csv_read(fixed/"threshold_metrics.csv", [
        "dataset", "model", "seed", "protected_attribute", "operating_threshold",
        "n_test", "n_group0", "n_group1", "n_min_group"
    ], keys=["dataset", "model", "seed", "protected_attribute", "operating_threshold"])
    manifest_path = fixed/"run_manifest.csv"
    if manifest_path.exists():
        manifest = csv_read(manifest_path, ["dataset", "model", "seed", "n_test", "fixed_test_seed"],
                            keys=["dataset", "model", "seed"])
        check(manifest["fixed_test_seed"].nunique() == 1,
              "manifest uses one fixed-test partition seed", log)
        merged = metrics.merge(manifest[["dataset", "model", "seed", "n_test"]],
                               on=["dataset", "model", "seed"], how="left",
                               suffixes=("_metrics", "_manifest"), validate="many_to_one")
        check(merged["n_test_manifest"].notna().all() and
              (merged["n_test_metrics"] == merged["n_test_manifest"]).all(),
              "metric test sizes match run manifest", log)
        check(len(manifest) == metrics[["dataset", "model", "seed"]].drop_duplicates().shape[0],
              "all metric seed/model combinations are in the manifest", log)
    else:
        print("  [INFO] run_manifest.csv missing; skipping seed check")
    # Fixed test labels/attributes must have identical group counts per dataset+attribute
    # across ALL seeds AND all models, not just a constant total n_test.
    group_keys = ["dataset", "protected_attribute"]
    for col in ("n_test", "n_group0", "n_group1", "n_min_group",
                "n_positive_group0", "n_positive_group1", "n_negative_group0", "n_negative_group1"):
        if col in metrics:
            check((metrics.groupby(group_keys)[col].nunique(dropna=False) == 1).all(),
                  f"fixed test has invariant {col} by dataset/attribute", log)
    num_seeds = metrics.groupby(["dataset", "model"])["seed"].nunique()
    print("  Model-fit counts by dataset/model: min",num_seeds.min(),"max",num_seeds.max())
    check((num_seeds == num_seeds.max()).all(), "same number of seeds in each dataset/model", log)

    e1 = csv_read(fixed/"experiment1_retraining_reproducibility.csv",
                  E1KEY + ["pass_probability", "n_retrainings", "verdict_instability"], keys=E1KEY)
    e3 = csv_read(fixed/"experiment3_audit_stability_surface.csv",
                  KEY + ["pass_probability", "n_retrainings", "verdict_instability"], keys=KEY)
    check_surface(e3, "fixed surface", log)
    p1 = e3[np.isclose(e3["operating_threshold"], t0)].drop(columns="operating_threshold")
    e1_e3 = e1.merge(p1, on=E1KEY, how="outer", suffixes=("_e1", "_e3"), indicator=True,
                     validate="one_to_one")
    check((e1_e3["_merge"] == "both").all(), "Experiment 1 matches surface at primary threshold", log)
    if (e1_e3["_merge"] == "both").all():
        for field in ("pass_probability", "n_retrainings", "verdict_instability"):
            check(np.allclose(e1_e3[f"{field}_e1"], e1_e3[f"{field}_e3"], atol=1e-9),
                  f"Exp1 versus Exp3 {field} agrees", log)
    n_by_family = e3.groupby(FAMILY)["n_retrainings"].nunique()
    check((n_by_family == 1).all(), "valid retraining counts constant across t,tau within audit family", log)
    return e3


def interval_table(surface, out, epsilons=(0.05,0.10,0.20)):
    print("\n[2] Uncertainty and epsilon sensitivity")
    z = float(norm.ppf(0.975))
    n = surface["n_retrainings"].to_numpy(dtype=int)
    k = np.rint(n*surface["pass_probability"].to_numpy(float)).astype(int)
    phat = k/n
    denom = 1 + z*z/n
    center = (phat + z*z/(2*n)) / denom
    half = z*np.sqrt(phat*(1-phat)/n + z*z/(4*n*n))/denom
    low, high = np.maximum(0, center-half), np.minimum(1, center+half)
    # VI(p)=2p(1-p) concave, maximum at 1/2, so its CI endpoints
    # are NOT obtained by sorting VI(low) and VI(high) alone.
    vl, vh = 2*low*(1-low), 2*high*(1-high)
    vi_lo = np.minimum(vl,vh)
    vi_hi = np.where((low <= 0.5) & (high >= 0.5), 0.5, np.maximum(vl,vh))
    d = surface[KEY + ["n_retrainings", "pass_probability", "verdict_instability"]].copy()
    d["pass_count"] = k
    d["p_wilson_low"] = low
    d["p_wilson_high"] = high
    d["vi_interval_low"] = vi_lo
    d["vi_interval_high"] = vi_hi
    d["vi_unbiased_estimate"] = 2*k*(n-k)/(n*(n-1))
    results=[]
    for eps in epsilons:
        label = np.where(phat <= eps, "stable_outside",
                         np.where(phat >= 1-eps,"stable_within","unstable"))
        certain = np.where(high <= eps, "stable_outside",
                    np.where(low >= 1-eps, "stable_within",
                    np.where((low > eps) & (high < 1-eps), "unstable", "uncertain")))
        col = f"region_eps_{str(eps).replace('.','p')}"
        d[col] = label
        d[f"certainty_eps_{str(eps).replace('.','p')}"] = certain
        results.append({"epsilon":eps,"n_points":len(d),
                        "unstable_n":int((label=="unstable").sum()),
                        "unstable_share":float((label=="unstable").mean()),
                        "uncertain_n":int((certain=="uncertain").sum()),
                        "uncertain_share":float((certain=="uncertain").mean()),
                        "reliably_unstable_n":int((certain=="unstable").sum())})
    save_csv(d,out/"uncertainty_by_surface_point.csv")
    save_csv(pd.DataFrame(results),out/"epsilon_and_uncertainty_summary.csv")
    print(pd.DataFrame(results).round(4).to_string(index=False))


def summarize_matched(g, label):
    x = g["verdict_instability_original"].to_numpy(float)
    y = g["verdict_instability_fixed"].to_numpy(float)
    return {"group":label,"n_matched":len(g),
            "original_mean_vi":float(x.mean()), "fixed_mean_vi":float(y.mean()),
            "delta_vi_fixed_minus_original":float((y-x).mean()),
            "original_nonzero_share":float((x>1e-12).mean()),
            "fixed_nonzero_share":float((y>1e-12).mean()),
            "delta_nonzero_share":float((y>1e-12).mean()-(x>1e-12).mean()),
            "original_unstable_share":float(g["unstable_original"].mean()),
            "fixed_unstable_share":float(g["unstable_fixed"].mean()),
            "delta_unstable_share":float((g["unstable_fixed"]-g["unstable_original"]).mean())}


def bootstrap_matched(matched, nboot, seed, out):
    if nboot<=0: return
    # Sample whole dataset-model blocks together. This preserves repeated
    # tolerance and attribute measurements rather than treating rows as iid.
    grp = matched.groupby(["dataset", "model"], sort=True)
    blocks=[]
    for _, sub in grp:
        blocks.append((len(sub), float(sub["vi_delta"].sum()),
                       float(sub["unstable_delta"].sum()),
                       float(sub["nonzero_delta"].sum())))
    blocks=np.asarray(blocks, dtype=float)
    if len(blocks)<2:
        print("  [SKIP] fewer than 2 dataset/model blocks for bootstrap")
        return
    rng=np.random.default_rng(seed)
    vals=np.empty((nboot,3),float)
    for i in range(nboot):
        sample = blocks[rng.integers(0,len(blocks),len(blocks))]
        vals[i] = sample[:,1:].sum(axis=0)/sample[:,0].sum()
    names=["delta_vi_fixed_minus_original", "delta_unstable_share", "delta_nonzero_share"]
    rows=[]
    for j,name in enumerate(names):
        lo,hi=np.percentile(vals[:,j],[2.5,97.5])
        rows.append({"statistic":name,"bootstrap_low_2p5":lo,
                     "bootstrap_high_97p5":hi,"n_dataset_model_blocks":len(blocks),
                     "bootstrap_repetitions":nboot})
    save_csv(pd.DataFrame(rows),out/"matched_dataset_model_block_bootstrap.csv")
    print("  Bootstrap is descriptive: dataset/model blocks are not a random sample of all applications.")


def matched_comparison(original_dir,fixed_surface,out,eps,t0,boot,seed,require,log):
    original_path=original_dir/"experiment3_audit_stability_surface.csv"
    print("\n[3] Matched original vs fixed-test comparison")
    if not original_path.is_file():
        if require: raise FileNotFoundError(original_path)
        print(f"  [SKIP] Original surface missing: {original_path}")
        print("  Re-run with --original-dir pointing to original results.")
        return
    orig=csv_read(original_path,KEY+["pass_probability","n_retrainings","verdict_instability"],keys=KEY)
    check_surface(orig,"original surface",log)
    a=orig[KEY+["pass_probability","n_retrainings","verdict_instability"]]
    b=fixed_surface[KEY+["pass_probability","n_retrainings","verdict_instability"]]
    outer=a.merge(b,on=KEY,how="outer",suffixes=("_original","_fixed"),indicator=True,
                  validate="one_to_one")
    join_ct=outer["_merge"].value_counts()
    save_csv(pd.DataFrame({"match_status":join_ct.index.astype(str),"n":join_ct.to_numpy()}),
             out/"matched_coverage.csv")
    unmatched = outer.loc[outer["_merge"] != "both", KEY+["_merge"]].copy()
    save_csv(unmatched, out/"unmatched_surface_keys.csv")
    m=outer.loc[outer["_merge"]=="both"].copy()
    if m.empty: raise RuntimeError("Original and fixed surfaces have no matching configurations")
    m["unstable_original"] = ((m["pass_probability_original"]>eps)&
                               (m["pass_probability_original"]<1-eps)).astype(int)
    m["unstable_fixed"] = ((m["pass_probability_fixed"]>eps)&
                            (m["pass_probability_fixed"]<1-eps)).astype(int)
    m["vi_delta"]=m["verdict_instability_fixed"]-m["verdict_instability_original"]
    m["unstable_delta"]=m["unstable_fixed"]-m["unstable_original"]
    m["nonzero_delta"]=(m["verdict_instability_fixed"]>1e-12).astype(int)-(
                       m["verdict_instability_original"]>1e-12).astype(int)
    save_csv(m.drop(columns="_merge"),out/"matched_surface_points.csv")
    summaries=[summarize_matched(m,"all thresholds")]
    primary=m[np.isclose(m["operating_threshold"],t0)]
    if len(primary): summaries.append(summarize_matched(primary,f"primary t={t0}"))
    for dataset,sub in m.groupby("dataset",sort=True):
        summaries.append(summarize_matched(sub, f"dataset: {dataset}"))
    for metric,sub in m.groupby("fairness_metric",sort=True):
        summaries.append(summarize_matched(sub, f"metric: {metric}"))
    save_csv(pd.DataFrame(summaries),out/"matched_summary.csv")
    family = m.groupby(FAMILY).agg(n_points=("vi_delta","size"),
             mean_delta_vi=("vi_delta","mean"),
             mean_delta_unstable=("unstable_delta","mean"),
             mean_delta_nonzero=("nonzero_delta","mean")).reset_index()
    save_csv(family,out/"matched_audit_family_effects.csv")
    bootstrap_matched(m,boot,seed,out)
    check(len(m)==int(join_ct.get("both",0)),"one-to-one matched surface points",log)
    print("  Matched points:",len(m),"; original only:",int(join_ct.get("left_only",0)),
          "; fixed only:",int(join_ct.get("right_only",0)))
    print(pd.DataFrame(summaries).head(2).round(5).to_string(index=False))


def parse_stored_coefficient(path, variable):
    if not path.is_file(): return np.nan
    text=path.read_text(encoding="utf-8")
    pat=rf"^\s*{re.escape(variable)}\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s+"
    match=re.search(pat,text,re.MULTILINE)
    return float(match.group(1)) if match else np.nan


def fit_regressions(fixed,out,log):
    print("\n[4] Regression replication, dependence sensitivity, dataset exclusion")
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise ImportError("Install required packages: python -m pip install statsmodels scipy pandas numpy") from exc
    fname=fixed/"experiment4_predictive_vs_audit_stability.csv"
    needed=E1KEY+["p","verdict_instability","sd_accuracy","sd_auroc",
                 "effective_support_n","log_effective_support_n",
                 "standardized_boundary_distance","vi_scaled","unstable_region","audit_family"]
    reg=csv_read(fname,needed,keys=E1KEY)
    # Check and report nesting/pseudo-replication.
    fam_expected=reg[FAMILY].astype(str).agg("__".join,axis=1)
    check((fam_expected==reg["audit_family"]).all(),"audit-family identifiers match grouping keys",log)
    check(np.allclose(reg["vi_scaled"],2*reg["verdict_instability"]),"2VI scaled outcome",log)
    check(np.allclose(reg["verdict_instability"],2*reg["p"]*(1-reg["p"])),"regression VI identity",log)
    sdcounts=reg.groupby(["dataset","model"])["sd_accuracy"].nunique()
    check((sdcounts==1).all(),"sd_accuracy repeats within dataset/model blocks",log)
    block_n=reg[["dataset","model"]].drop_duplicates().shape[0]
    print(f"  Rows={len(reg)}, audit families={reg.audit_family.nunique()}, dataset/model blocks={block_n}")
    reg=reg.replace([np.inf,-np.inf],np.nan).dropna(
        subset=["verdict_instability","sd_accuracy","sd_auroc","effective_support_n",
                "log_effective_support_n","audit_family"])
    reg_b=reg.dropna(subset=["standardized_boundary_distance"]).copy()
    formula1=("verdict_instability ~ log_effective_support_n + sd_accuracy + "
              "C(model) + C(fairness_metric) + C(dataset)")
    formula2=("verdict_instability ~ log_effective_support_n + "
              "standardized_boundary_distance + sd_accuracy + "
              "C(model) + C(fairness_metric) + C(dataset)")
    form_base=(" ~ log_effective_support_n + standardized_boundary_distance + "
               "sd_accuracy + C(model) + C(fairness_metric) + C(dataset)")
    reg["dataset_model_cluster"]=reg["dataset"].astype(str)+"__"+reg["model"].astype(str)
    reg_b["dataset_model_cluster"]=reg_b["dataset"].astype(str)+"__"+reg_b["model"].astype(str)

    rows=[]
    comp=[]
    diagnostics=[]
    def record(fit, name, clustering, data):
        for param in fit.params.index:
            ci=fit.conf_int().loc[param]
            rows.append({"analysis":name,"clustering":clustering,"parameter":param,
                         "estimate":float(fit.params[param]),
                         "std_error":float(fit.bse[param]),
                         "p_value":float(fit.pvalues[param]),
                         "ci_low":float(ci.iloc[0]),"ci_high":float(ci.iloc[1]),
                         "n_obs":int(fit.nobs),
                         "n_audit_families":data["audit_family"].nunique(),
                         "n_dataset_models":data["dataset_model_cluster"].nunique(),
                         "r_squared":float(fit.rsquared) if hasattr(fit,"rsquared") else np.nan})
        return fit

    for cluster in ("audit_family","dataset_model_cluster"):
        for nm,formula,data in (("ols_contextual",formula1,reg),
                                ("ols_adjusted",formula2,reg_b)):
            try:
                fit=smf.ols(formula,data=data).fit(cov_type="cluster",cov_kwds={"groups":data[cluster]})
                record(fit,nm,cluster,data)
                diagnostics.append({"analysis": nm, "clustering": cluster,
                                    "diagnostic": "ols_condition_number",
                                    "value": float(np.linalg.cond(fit.model.exog)),
                                    "warning": "high if >1000; depends on feature scaling"})
            except Exception as exc:
                print("  [WARN]", nm, cluster, exc)
                comp.append({"analysis":nm,"clustering":cluster,"error":str(exc)})

    gee_models=(("gee_fractional_exchangeable","vi_scaled","Exchangeable"),
                ("gee_binary_exchangeable","unstable_region","Exchangeable"),
                ("gee_fractional_independence","vi_scaled","Independence"),
                ("gee_binary_independence","unstable_region","Independence"))
    for nm,y,covtype in gee_models:
        try:
            cov=sm.cov_struct.Exchangeable() if covtype=="Exchangeable" else sm.cov_struct.Independence()
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                fit=smf.gee(y+form_base,groups="audit_family",data=reg_b,
                            family=sm.families.Binomial(),cov_struct=cov).fit(maxiter=200)
            record(fit,nm,"audit_family_"+covtype.lower(),reg_b)
            predictions = np.asarray(fit.predict())
            diagnostics.append({"analysis":nm,"clustering":covtype,
                                "diagnostic":"fitted_probability_near_0_or_1_share",
                                "value":float(np.mean((predictions<1e-6)|(predictions>1-1e-6))),
                                "warning":"near-boundary predictions may indicate quasi-separation"})
            status={"analysis":nm,"converged":str(getattr(fit,"converged","unknown")),
                    "iterations":len(fit.fit_history.get("params",[])),
                    "warning_count":len(w),"warning_text":"; ".join(str(x.message) for x in w)[:500]}
            comp.append(status)
            if w: print(f"  [WARN] {nm}: {len(w)} warnings; see regression_fit_status.csv")
        except Exception as exc:
            comp.append({"analysis":nm,"error":str(exc)})
            print("  [WARN]",nm,exc)

    if not rows:
        raise RuntimeError("No regression model was estimated")
    estimates=pd.DataFrame(rows)
    save_csv(estimates,out/"regression_coefficients_and_standard_errors.csv")
    save_csv(pd.DataFrame(comp),out/"regression_fit_status.csv")
    save_csv(pd.DataFrame(diagnostics),out/"regression_numerical_diagnostics.csv")

    # Reproduction check: compare original saved regression TXT (rounded to 4 decimals)
    # against exact rerun coefficients, to detect stale analysis files.
    comparisons=[]
    for nm,stored in ESTIMATORS.items():
        ref=estimates[(estimates.analysis==nm) &
                      (estimates.clustering.str.startswith("audit_family"))]
        for _,r in ref.iterrows():
            old=parse_stored_coefficient(fixed/stored,r["parameter"])
            tol=max(0.00011,abs(old)*0.00003) if np.isfinite(old) else np.nan
            comparisons.append({"analysis":nm,"parameter":r["parameter"],
                                "saved_rounded_coefficient":old,
                                "recomputed_coefficient":r["estimate"],
                                "difference":r["estimate"]-old if np.isfinite(old) else np.nan,
                                "matches_saved_to_printed_precision":bool(abs(r["estimate"]-old)<=tol)
                                if np.isfinite(old) else False,
                                "source_text_exists":(fixed/stored).exists()})
    rec=pd.DataFrame(comparisons)
    save_csv(rec,out/"saved_vs_recomputed_regressions.csv")
    if len(rec):
        mismatches=(~rec.matches_saved_to_printed_precision).sum()
        check(mismatches==0,f"saved summary text coefficients align with refits ({mismatches} mismatches)",log)

    # Leave-one-dataset-out for OLS to expose dependence on individual benchmarks.
    lodo=[]
    for omit in sorted(reg_b.dataset.unique()):
        sub=reg_b[reg_b.dataset!=omit]
        for nm,formula in (("ols_contextual",formula1),("ols_adjusted",formula2)):
            try:
                fit=smf.ols(formula,data=sub).fit(cov_type="cluster",cov_kwds={"groups":sub["audit_family"]})
                for par in ("log_effective_support_n","sd_accuracy","standardized_boundary_distance"):
                    if par in fit.params:
                        lodo.append({"excluded_dataset":omit,"analysis":nm,"parameter":par,
                                     "estimate":fit.params[par],"p_value":fit.pvalues[par],
                                     "n_obs":len(sub),"n_audit_families":sub.audit_family.nunique()})
            except Exception as exc:
                print("  [WARN] LODO",omit,nm,exc)
    save_csv(pd.DataFrame(lodo),out/"leave_one_dataset_out_ols.csv")
    important=estimates[(estimates["parameter"].isin(["sd_accuracy","log_effective_support_n",
                                                       "standardized_boundary_distance"]))]
    print("\n  Key coefficient sensitivity:\n",important[["analysis","clustering","parameter","estimate","p_value"]].round(4).to_string(index=False))


def main():
    args=arguments()
    if not 0<args.epsilon<0.5: raise ValueError("--epsilon must be between 0 and 0.5")
    if args.bootstrap <0: raise ValueError("--bootstrap must be >= 0")
    args.out.mkdir(parents=True,exist_ok=True)
    fixed_root=args.fixed_dir.resolve()
    out_root=args.out.resolve()
    if fixed_root == out_root or fixed_root in out_root.parents:
        # A validation subdirectory in fixed data can be okay, but not a matching root
        if fixed_root==out_root:
            raise ValueError("--out cannot be identical to --fixed-dir")
    if args.original_dir.resolve()==out_root:
        raise ValueError("--out cannot be identical to --original-dir")
    log=[]
    surface=validate_fixed_inputs(args.fixed_dir,args.primary_t,args.out,log)
    interval_table(surface,args.out)
    matched_comparison(args.original_dir,surface,args.out,args.epsilon,args.primary_t,
                       args.bootstrap,args.seed,args.require_original,log)
    if not args.skip_regressions:
        fit_regressions(args.fixed_dir,args.out,log)
    save_csv(pd.DataFrame(log),args.out/"validation_checks.csv")
    metadata={"original_dir":str(args.original_dir),"fixed_dir":str(args.fixed_dir),
              "epsilon":args.epsilon,"primary_t":args.primary_t,
              "bootstrap_repetitions":args.bootstrap,"random_seed":args.seed,
              "checks_failed":int(sum(x["status"]=="FAIL" for x in log)),
              "notes":["This is an observational protocol comparison, not a causal decomposition.",
                       "Tolerance-expanded rows are dependent; audit-family clusters also share datasets and models.",
                       "Dataset-model bootstrap intervals are descriptive with few blocks.",
                       "Wilson intervals are pointwise conditional on independent sampled realizations.",
                       "Boundary-distance predictors are mathematically related to the outcome."]}
    (args.out/"validation_manifest.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(f"\n[done] Outputs: {args.out}")
    print("[done] Failed checks:",metadata["checks_failed"])
    if metadata["checks_failed"]:
        raise SystemExit("Validation checks failed; inspect validation_checks.csv")


if __name__=="__main__":
    main()
