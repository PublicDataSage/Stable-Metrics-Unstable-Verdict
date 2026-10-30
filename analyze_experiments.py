from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import yaml

from src.metrics import FAIRNESS_VIOLATION_COLUMNS, FAIRNESS_VALIDITY_COLUMNS


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    return ap.parse_args()


def grid(start, stop, step):
    return np.round(
        np.arange(start, stop + step / 2, step),
        10
    )


def add_audit_rows(metrics, cfg):
    diff_tau = grid(
        **cfg["audit_tolerances"]["difference_metrics"]
    )

    ratio_tau = grid(
        **cfg["audit_tolerances"]["ratio_metric"]
    )

    # DPR config is expressed as a minimum acceptable ratio.
    # Convert to violation tolerance:
    #
    #   V = 1 - DPR
    #
    # so all metrics use:
    #
    #   PASS <=> violation <= audit_tolerance
    #
    ratio_violation_tau = np.sort(
        np.unique(1.0 - ratio_tau)
    )

    parts = []

    for metric in FAIRNESS_VIOLATION_COLUMNS:
        taus = (
            ratio_violation_tau
            if metric == "dpr_violation"
            else diff_tau
        )

        validity_col = FAIRNESS_VALIDITY_COLUMNS[metric]

        base_cols = [
            "dataset",
            "model",
            "seed",
            "protected_attribute",
            "operating_threshold",
            "n_test",
            "n_min_group",
            "n_min_positive_group",
            "n_min_negative_group",
            "accuracy",
            "auroc",
            "f1",
            metric,
            validity_col,
        ]

        tmp = metrics[base_cols].rename(
            columns={
                metric: "violation",
                validity_col: "analysis_valid",
            }
        )

        tmp = tmp[
            tmp["analysis_valid"].fillna(False)
            & tmp["violation"].notna()
        ].copy()

        if tmp.empty:
            continue

        repeated = tmp.loc[
            tmp.index.repeat(len(taus))
        ].copy()

        repeated["audit_tolerance"] = np.tile(
            taus,
            len(tmp)
        )

        repeated["fairness_metric"] = metric

        repeated["pass"] = (
            repeated["violation"]
            <= repeated["audit_tolerance"]
        ).astype(int)

        parts.append(repeated)

    if not parts:
        raise RuntimeError(
            "No valid audit rows were generated."
        )

    return pd.concat(
        parts,
        ignore_index=True
    )


def experiment1(audit, out_dir, primary_t):
    d = audit[
        np.isclose(
            audit["operating_threshold"],
            primary_t
        )
    ].copy()

    gcols = [
        "dataset",
        "model",
        "protected_attribute",
        "fairness_metric",
        "audit_tolerance",
    ]

    agg = (
        d.groupby(
            gcols,
            dropna=False
        )
        .agg(
            pass_probability=("pass", "mean"),
            n_retrainings=("seed", "nunique"),
            mean_violation=("violation", "mean"),
            sd_violation=("violation", "std"),
            median_violation=("violation", "median"),
            q05=(
                "violation",
                lambda x: x.quantile(.05)
            ),
            q95=(
                "violation",
                lambda x: x.quantile(.95)
            ),
            mean_accuracy=("accuracy", "mean"),
            sd_accuracy=("accuracy", "std"),
            mean_auroc=("auroc", "mean"),
            sd_auroc=("auroc", "std"),
            mean_min_group_n=("n_min_group", "mean"),
            mean_min_positive_group_n=(
                "n_min_positive_group",
                "mean"
            ),
            mean_min_negative_group_n=(
                "n_min_negative_group",
                "mean"
            ),
        )
        .reset_index()
    )

    p = agg["pass_probability"]

    agg["verdict_instability"] = (
        2 * p * (1 - p)
    )

    agg["audit_agreement"] = (
        1 - agg["verdict_instability"]
    )

    agg.to_csv(
        out_dir
        / "experiment1_retraining_reproducibility.csv",
        index=False
    )

    return agg


def experiment2(e1, out_dir):
    d = e1.copy()

    d["metric_iqr_proxy"] = (
        d["q95"] - d["q05"]
    )

    d["boundary_distance"] = np.abs(
        d["mean_violation"]
        - d["audit_tolerance"]
    )

    d["standardized_boundary_distance"] = (
        d["boundary_distance"]
        / d["sd_violation"].replace(0, np.nan)
    )

    d["boundary_inside_90pct_range"] = (
        (d["audit_tolerance"] >= d["q05"])
        &
        (d["audit_tolerance"] <= d["q95"])
    )

    q_spread = d["metric_iqr_proxy"].quantile(.25)
    q_vi = d["verdict_instability"].quantile(.75)

    d["low_spread_high_instability"] = (
        (d["metric_iqr_proxy"] <= q_spread)
        &
        (d["verdict_instability"] >= q_vi)
    )

    q_spread_high = d[
        "metric_iqr_proxy"
    ].quantile(.75)

    d["high_spread_zero_instability"] = (
        (d["metric_iqr_proxy"] >= q_spread_high)
        &
        (d["verdict_instability"] == 0)
    )

    d.to_csv(
        out_dir
        / "experiment2_metric_vs_verdict_stability.csv",
        index=False
    )

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)

    ax.scatter(
        d["metric_iqr_proxy"],
        d["verdict_instability"],
        alpha=.55
    )

    ax.set_xlabel(
        "Fairness metric 90% range (Q95-Q05)"
    )

    ax.set_ylabel(
        "Verdict instability 2p(1-p)"
    )

    ax.set_title(
        "Metric stability vs audit-verdict stability"
    )

    fig.tight_layout()

    fig.savefig(
        out_dir
        / "experiment2_metric_vs_verdict_stability.png",
        dpi=220
    )

    plt.close(fig)

    dd = d.dropna(
        subset=[
            "standardized_boundary_distance",
            "verdict_instability",
        ]
    )

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)

    ax.scatter(
        dd["standardized_boundary_distance"],
        dd["verdict_instability"],
        alpha=.50
    )

    ax.set_xlabel(
        "Standardized distance to audit boundary"
    )

    ax.set_ylabel(
        "Verdict instability 2p(1-p)"
    )

    ax.set_title(
        "Boundary proximity vs audit-verdict instability"
    )

    fig.tight_layout()

    fig.savefig(
        out_dir
        / "experiment2_boundary_distance_vs_instability.png",
        dpi=220
    )

    plt.close(fig)

    return d


def experiment3(audit, out_dir, epsilon):
    gcols = [
        "dataset",
        "model",
        "protected_attribute",
        "fairness_metric",
        "operating_threshold",
        "audit_tolerance",
    ]

    surf = (
        audit.groupby(gcols)
        .agg(
            pass_probability=("pass", "mean"),
            n_retrainings=("seed", "nunique"),
            mean_violation=("violation", "mean"),
        )
        .reset_index()
    )

    p = surf["pass_probability"]

    surf["verdict_instability"] = (
        2 * p * (1 - p)
    )

    surf["region"] = np.select(
        [
            p >= 1 - epsilon,
            p <= epsilon,
        ],
        [
            "stable_within_tolerance",
            "stable_outside_tolerance",
        ],
        default="unstable"
    )

    surf.to_csv(
        out_dir
        / "experiment3_audit_stability_surface.csv",
        index=False
    )

    # Heatmaps
    for keys, sub in surf.groupby(
        [
            "dataset",
            "model",
            "protected_attribute",
            "fairness_metric",
        ]
    ):
        pivot = (
            sub.pivot(
                index="audit_tolerance",
                columns="operating_threshold",
                values="pass_probability"
            )
            .sort_index()
            .sort_index(axis=1)
        )

        if pivot.empty:
            continue

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111)

        im = ax.imshow(
            pivot.values,
            origin="lower",
            aspect="auto",
            extent=[
                pivot.columns.min(),
                pivot.columns.max(),
                pivot.index.min(),
                pivot.index.max(),
            ],
            vmin=0,
            vmax=1,
        )

        ax.set_xlabel(
            "Operational threshold t"
        )

        ax.set_ylabel(
            "Audit tolerance tau"
        )

        ax.set_title(
            "Audit Stability Surface: "
            + " | ".join(map(str, keys))
        )

        fig.colorbar(
            im,
            ax=ax,
            label="Pass probability"
        )

        fig.tight_layout()

        safe = "__".join(
            str(x).replace("/", "_")
            for x in keys
        )

        fig.savefig(
            out_dir
            / f"surface__{safe}.png",
            dpi=200
        )

        plt.close(fig)

    sens_parts = []
    step_parts = []

    for keys, sub in surf.groupby(
        [
            "dataset",
            "model",
            "protected_attribute",
            "fairness_metric",
        ]
    ):
        pvt = (
            sub.pivot(
                index="audit_tolerance",
                columns="operating_threshold",
                values="pass_probability"
            )
            .sort_index()
            .sort_index(axis=1)
        )

        if (
            pvt.shape[0] < 2
            or pvt.shape[1] < 2
        ):
            continue

        tau_vals = pvt.index.to_numpy(float)
        t_vals = pvt.columns.to_numpy(float)
        z = pvt.to_numpy(float)

        dt = np.gradient(t_vals)
        dtau = np.gradient(tau_vals)

        osens = np.abs(
            np.gradient(
                z,
                axis=1
            )
            / dt[np.newaxis, :]
        )

        gsens = np.abs(
            np.gradient(
                z,
                axis=0
            )
            / dtau[:, np.newaxis]
        )

        rr, cc = np.meshgrid(
            tau_vals,
            t_vals,
            indexing="ij"
        )

        tmp = pd.DataFrame({
            "audit_tolerance":
                rr.ravel(),
            "operating_threshold":
                cc.ravel(),
            "operational_sensitivity":
                osens.ravel(),
            "governance_boundary_sensitivity":
                gsens.ravel(),
        })

        for name, val in zip(
            [
                "dataset",
                "model",
                "protected_attribute",
                "fairness_metric",
            ],
            keys
        ):
            tmp[name] = val

        sens_parts.append(tmp)

        op_step_change = np.full_like(
            z,
            np.nan,
            dtype=float
        )

        audit_step_change = np.full_like(
            z,
            np.nan,
            dtype=float
        )

        op_step_change[:, 1:] = np.abs(
            z[:, 1:]
            - z[:, :-1]
        )

        audit_step_change[1:, :] = np.abs(
            z[1:, :]
            - z[:-1, :]
        )

        step_tmp = pd.DataFrame({
            "audit_tolerance":
                rr.ravel(),
            "operating_threshold":
                cc.ravel(),
            "operational_step_change":
                op_step_change.ravel(),
            "audit_boundary_step_change":
                audit_step_change.ravel(),
        })

        step_tmp[
            "operating_threshold_step_size"
        ] = (
            np.median(
                np.diff(t_vals)
            )
            if len(t_vals) > 1
            else np.nan
        )

        step_tmp[
            "audit_tolerance_step_size"
        ] = (
            np.median(
                np.diff(tau_vals)
            )
            if len(tau_vals) > 1
            else np.nan
        )

        for name, val in zip(
            [
                "dataset",
                "model",
                "protected_attribute",
                "fairness_metric",
            ],
            keys
        ):
            step_tmp[name] = val

        step_parts.append(step_tmp)

    if sens_parts:
        pd.concat(
            sens_parts,
            ignore_index=True
        ).to_csv(
            out_dir
            / "experiment3_sensitivities.csv",
            index=False
        )

    if step_parts:
        step_df = pd.concat(
            step_parts,
            ignore_index=True
        )

        step_df.to_csv(
            out_dir
            / "experiment3_step_changes.csv",
            index=False
        )

        step_summary = (
            step_df.groupby(
                [
                    "dataset",
                    "fairness_metric",
                ],
                dropna=False
            )
            .agg(
                mean_operational_step_change=(
                    "operational_step_change",
                    "mean"
                ),
                median_operational_step_change=(
                    "operational_step_change",
                    "median"
                ),
                mean_audit_boundary_step_change=(
                    "audit_boundary_step_change",
                    "mean"
                ),
                median_audit_boundary_step_change=(
                    "audit_boundary_step_change",
                    "median"
                ),
            )
            .reset_index()
        )

        step_summary.to_csv(
            out_dir
            / "experiment3_step_change_summary.csv",
            index=False
        )

    return surf


def _effective_support(row):
    """
    Metric-specific effective subgroup support.

    DPD / DPR:
        minimum protected-group size

    EOD:
        minimum positive-label subgroup size

    Equalized odds:
        minimum of positive-label and negative-label
        subgroup support
    """
    metric = row["fairness_metric"]

    if metric in {
        "dpd_violation",
        "dpr_violation",
    }:
        return row["mean_min_group_n"]

    if metric == "eod_violation":
        return row[
            "mean_min_positive_group_n"
        ]

    if metric == "equalized_odds_violation":
        return min(
            row[
                "mean_min_positive_group_n"
            ],
            row[
                "mean_min_negative_group_n"
            ],
        )

    return row["mean_min_group_n"]


def _save_model_summary(
    fit,
    path
):
    path.write_text(
        fit.summary().as_text(),
        encoding="utf-8"
    )


def experiment4(
    metrics,
    audit,
    out_dir,
    primary_t,
    instability_epsilon
):
    # --------------------------------------------------
    # Predictive stability across retrainings
    # --------------------------------------------------

    m = metrics[
        np.isclose(
            metrics["operating_threshold"],
            primary_t
        )
    ].copy()

    perf = (
        m.groupby(
            [
                "dataset",
                "model",
                "seed",
            ]
        )
        .agg(
            accuracy=("accuracy", "first"),
            auroc=("auroc", "first"),
            f1=("f1", "first"),
        )
        .reset_index()
    )

    perf_summary = (
        perf.groupby(
            [
                "dataset",
                "model",
            ]
        )
        .agg(
            sd_accuracy=("accuracy", "std"),
            sd_auroc=("auroc", "std"),
            sd_f1=("f1", "std"),
            mean_accuracy=("accuracy", "mean"),
            mean_auroc=("auroc", "mean"),
        )
        .reset_index()
    )

    # --------------------------------------------------
    # Audit stability at primary threshold
    # --------------------------------------------------

    a = audit[
        np.isclose(
            audit["operating_threshold"],
            primary_t
        )
    ].copy()

    gcols = [
        "dataset",
        "model",
        "protected_attribute",
        "fairness_metric",
        "audit_tolerance",
    ]

    gov = (
        a.groupby(gcols)
        .agg(
            p=("pass", "mean"),
            n_retrainings=("seed", "nunique"),
            mean_violation=("violation", "mean"),
            sd_violation=("violation", "std"),
            mean_min_group_n=("n_min_group", "mean"),
            mean_min_positive_group_n=(
                "n_min_positive_group",
                "mean"
            ),
            mean_min_negative_group_n=(
                "n_min_negative_group",
                "mean"
            ),
        )
        .reset_index()
    )

    gov["verdict_instability"] = (
        2 * gov["p"] * (1 - gov["p"])
    )

    gov = gov.merge(
        perf_summary,
        on=[
            "dataset",
            "model",
        ],
        how="left"
    )

    # --------------------------------------------------
    # Audit family cluster identifier
    # --------------------------------------------------

    gov["audit_family"] = (
        gov["dataset"].astype(str)
        + "__"
        + gov["model"].astype(str)
        + "__"
        + gov[
            "protected_attribute"
        ].astype(str)
        + "__"
        + gov[
            "fairness_metric"
        ].astype(str)
    )

    # --------------------------------------------------
    # Metric-specific effective support
    # --------------------------------------------------

    gov["effective_support_n"] = (
        gov.apply(
            _effective_support,
            axis=1
        )
    )

    gov[
        "log_effective_support_n"
    ] = np.log1p(
        gov["effective_support_n"]
    )

    # --------------------------------------------------
    # Boundary-proximity mechanism
    # --------------------------------------------------

    gov["boundary_distance"] = np.abs(
        gov["mean_violation"]
        - gov["audit_tolerance"]
    )

    gov[
        "standardized_boundary_distance"
    ] = (
        gov["boundary_distance"]
        / gov[
            "sd_violation"
        ].replace(0, np.nan)
    )

    # --------------------------------------------------
    # Scaled bounded outcome
    #
    # VI is in [0, 0.5].
    # Scaling by 2 maps it to [0, 1].
    # --------------------------------------------------

    gov["vi_scaled"] = (
        2 * gov["verdict_instability"]
    )

    # --------------------------------------------------
    # Governance-oriented binary outcome
    # --------------------------------------------------

    eps = float(
        instability_epsilon
    )

    gov["unstable_region"] = (
        (gov["p"] > eps)
        &
        (gov["p"] < 1 - eps)
    ).astype(int)

    gov.to_csv(
        out_dir
        / "experiment4_predictive_vs_audit_stability.csv",
        index=False
    )

    # --------------------------------------------------
    # Analysis data
    # --------------------------------------------------

    reg = gov.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna(
        subset=[
            "verdict_instability",
            "effective_support_n",
            "log_effective_support_n",
            "sd_accuracy",
            "sd_auroc",
            "audit_family",
        ]
    ).copy()

    # ==================================================
    # MODEL A:
    # Contextual drivers only
    #
    # Clustered OLS
    # ==================================================

    formula_a = (
        "verdict_instability ~ "
        "log_effective_support_n + "
        "sd_accuracy + "
        "C(model) + "
        "C(fairness_metric) + "
        "C(dataset)"
    )

    fit_a = smf.ols(
        formula_a,
        data=reg
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups":
                reg["audit_family"]
        }
    )

    _save_model_summary(
        fit_a,
        out_dir
        / "experiment4_clustered_ols_contextual.txt"
    )

    # ==================================================
    # MODEL B:
    # Mechanism-adjusted clustered OLS
    #
    # Adds standardized distance to the audit boundary.
    # ==================================================

    reg_b = reg.dropna(
        subset=[
            "standardized_boundary_distance"
        ]
    ).copy()

    formula_b = (
        "verdict_instability ~ "
        "log_effective_support_n + "
        "standardized_boundary_distance + "
        "sd_accuracy + "
        "C(model) + "
        "C(fairness_metric) + "
        "C(dataset)"
    )

    fit_b = smf.ols(
        formula_b,
        data=reg_b
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups":
                reg_b["audit_family"]
        }
    )

    _save_model_summary(
        fit_b,
        out_dir
        / "experiment4_clustered_ols_mechanism_adjusted.txt"
    )

    # ==================================================
    # ROBUSTNESS MODEL C:
    # Fractional-response GEE
    #
    # Outcome:
    #   vi_scaled = 2 * VI
    #
    # This keeps the outcome in [0,1] and accounts
    # for within-audit-family dependence.
    # ==================================================

    gee_frac = smf.gee(
        (
            "vi_scaled ~ "
            "log_effective_support_n + "
            "standardized_boundary_distance + "
            "sd_accuracy + "
            "C(model) + "
            "C(fairness_metric) + "
            "C(dataset)"
        ),
        groups="audit_family",
        data=reg_b,
        family=sm.families.Binomial(),
        cov_struct=(
            sm.cov_struct.Exchangeable()
        ),
    ).fit()

    _save_model_summary(
        gee_frac,
        out_dir
        / "experiment4_fractional_gee.txt"
    )

    # ==================================================
    # ROBUSTNESS MODEL D:
    # Binary unstable-region GEE
    #
    # Outcome:
    #   1 if epsilon < p < 1-epsilon
    # ==================================================

    gee_binary = smf.gee(
        (
            "unstable_region ~ "
            "log_effective_support_n + "
            "standardized_boundary_distance + "
            "sd_accuracy + "
            "C(model) + "
            "C(fairness_metric) + "
            "C(dataset)"
        ),
        groups="audit_family",
        data=reg_b,
        family=sm.families.Binomial(),
        cov_struct=(
            sm.cov_struct.Exchangeable()
        ),
    ).fit()

    _save_model_summary(
        gee_binary,
        out_dir
        / "experiment4_binary_unstable_gee.txt"
    )

    # --------------------------------------------------
    # Model comparison table
    # --------------------------------------------------

    comparison = pd.DataFrame({
        "model": [
            "Clustered OLS - contextual",
            "Clustered OLS - mechanism adjusted",
            "Fractional GEE",
            "Binary unstable-region GEE",
        ],
        "n_observations": [
            int(fit_a.nobs),
            int(fit_b.nobs),
            int(gee_frac.nobs),
            int(gee_binary.nobs),
        ],
        "n_audit_families": [
            int(reg["audit_family"].nunique()),
            int(reg_b["audit_family"].nunique()),
            int(reg_b["audit_family"].nunique()),
            int(reg_b["audit_family"].nunique()),
        ],
    })

    comparison.to_csv(
        out_dir
        / "experiment4_model_comparison.csv",
        index=False
    )

    # --------------------------------------------------
    # Correlation diagnostics
    # --------------------------------------------------

    corr_cols = [
        "verdict_instability",
        "sd_accuracy",
        "sd_auroc",
        "sd_f1",
        "effective_support_n",
        "log_effective_support_n",
        "boundary_distance",
        "standardized_boundary_distance",
    ]

    corr = (
        gov[corr_cols]
        .corr(
            method="spearman"
        )
    )

    corr.to_csv(
        out_dir
        / "experiment4_spearman_correlations.csv"
    )

    # --------------------------------------------------
    # Predictive stability vs audit stability plot
    # --------------------------------------------------

    fig = plt.figure(
        figsize=(7, 5)
    )

    ax = fig.add_subplot(111)

    ax.scatter(
        gov["sd_accuracy"],
        gov["verdict_instability"],
        alpha=.55
    )

    ax.set_xlabel(
        "SD of accuracy across retrainings"
    )

    ax.set_ylabel(
        "Verdict instability"
    )

    ax.set_title(
        "Predictive stability vs audit-verdict stability"
    )

    fig.tight_layout()

    fig.savefig(
        out_dir
        / "experiment4_predictive_vs_audit_stability.png",
        dpi=220
    )

    plt.close(fig)

    # --------------------------------------------------
    # Effective support vs instability plot
    # --------------------------------------------------

    fig = plt.figure(
        figsize=(7, 5)
    )

    ax = fig.add_subplot(111)

    ax.scatter(
        gov["effective_support_n"],
        gov["verdict_instability"],
        alpha=.45
    )

    ax.set_xscale("log")

    ax.set_xlabel(
        "Metric-specific effective subgroup support"
    )

    ax.set_ylabel(
        "Verdict instability"
    )

    ax.set_title(
        "Effective subgroup support vs audit instability"
    )

    fig.tight_layout()

    fig.savefig(
        out_dir
        / "experiment4_effective_support_vs_instability.png",
        dpi=220
    )

    plt.close(fig)

    return gov


def main():
    args = parse_args()

    cfg = yaml.safe_load(
        open(
            args.config,
            "r",
            encoding="utf-8"
        )
    )

    out_dir = Path(
        cfg.get(
            "output_dir",
            "results"
        )
    )

    metrics_path = (
        out_dir
        / "threshold_metrics.csv"
    )

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Could not find {metrics_path}. "
            "Run run_experiments.py first."
        )

    metrics = pd.read_csv(
        metrics_path
    )

    audit = add_audit_rows(
        metrics,
        cfg
    )

    audit.to_csv(
        out_dir
        / "audit_long.csv",
        index=False
    )

    t0 = float(
        cfg.get(
            "primary_operating_threshold",
            .5
        )
    )

    eps = float(
        cfg["analysis"].get(
            "instability_epsilon",
            .1
        )
    )

    e1 = experiment1(
        audit,
        out_dir,
        t0
    )

    experiment2(
        e1,
        out_dir
    )

    experiment3(
        audit,
        out_dir,
        eps
    )

    experiment4(
        metrics,
        audit,
        out_dir,
        t0,
        eps
    )

    print(
        "Analysis complete."
    )

    print(
        "\nKey CSV outputs:"
    )

    for p in sorted(
        out_dir.glob(
            "experiment*.csv"
        )
    ):
        print(
            " ",
            p
        )

    print(
        "\nExperiment 4 model outputs:"
    )

    for name in [
        "experiment4_clustered_ols_contextual.txt",
        "experiment4_clustered_ols_mechanism_adjusted.txt",
        "experiment4_fractional_gee.txt",
        "experiment4_binary_unstable_gee.txt",
        "experiment4_model_comparison.csv",
        "experiment4_spearman_correlations.csv",
    ]:
        print(
            " ",
            out_dir / name
        )


if __name__ == "__main__":
    main()
