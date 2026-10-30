from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =========================================================
# PUBLICATION LAYOUT CONSTANTS
# =========================================================
# ACM two-column papers are roughly 7--7.2 inches across both columns.
# Keeping figures close to their final physical size prevents LaTeX from
# shrinking the text after export.
DOUBLE_COL_WIDTH = 7.15
SINGLE_COL_WIDTH = 3.40

# Colorblind-safe Okabe--Ito palette.
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#222222",
    "gray": "#777777",
    "lightgray": "#D9D9D9",
}


# =========================================================
# BASIC UTILITIES
# =========================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Generate cleaned publication-ready figures for the fairness audit stability study."
    )
    ap.add_argument(
        "--results",
        default="results",
        help="Directory containing experiment CSV files"
    )
    ap.add_argument(
        "--out",
        default="final_figures_clean",
        help="Output directory for final figures"
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG DPI"
    )
    return ap.parse_args()


def require(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def setup_style():
    """Large, clean typography designed to survive ACM two-column scaling."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 12.5,
        "axes.titlesize": 15.0,
        "axes.titleweight": "semibold",
        "axes.labelsize": 13.5,
        "axes.labelweight": "medium",
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "legend.fontsize": 11.0,
        "legend.title_fontsize": 11.5,
        "figure.titlesize": 15.5,
        "axes.linewidth": 1.2,
        "lines.linewidth": 1.8,
        "lines.markersize": 7.0,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "xtick.major.size": 4.5,
        "ytick.major.size": 4.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def polish_axes(ax, grid=True):
    """Consistent publication styling for ordinary Cartesian axes."""
    ax.tick_params(axis="both", which="major", pad=4)
    if grid:
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.25)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(1.2)


def save_figure(fig, out_dir: Path, stem: str, dpi: int):
    fig.savefig(
        out_dir / f"{stem}.png",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.06,
        facecolor="white",
    )
    fig.savefig(
        out_dir / f"{stem}.pdf",
        bbox_inches="tight",
        pad_inches=0.06,
        facecolor="white",
    )
    plt.close(fig)


def label_metric(metric: str) -> str:
    return {
        "dpd_violation": "DPD",
        "dpr_violation": "DPR",
        "eod_violation": "EOD",
        "equalized_odds_violation": "Equalized Odds",
    }.get(metric, metric)


def label_dataset(dataset: str) -> str:
    return {
        "adult": "Adult",
        "acs_income": "ACSIncome",
        "compas": "COMPAS",
        "german": "German Credit",
    }.get(dataset, dataset)


def short_model(model: str) -> str:
    return {
        "logistic_regression": "LR",
        "random_forest": "RF",
        "hist_gradient_boosting": "HGB",
    }.get(model, model)


def make_unstable_flag(df: pd.DataFrame, epsilon: float = 0.10):
    if "unstable_region" in df.columns:
        return df["unstable_region"].astype(int)

    if "p" in df.columns:
        return ((df["p"] > epsilon) & (df["p"] < 1 - epsilon)).astype(int)

    raise ValueError(
        "Experiment 4 CSV must contain either 'unstable_region' or 'p'."
    )


# =========================================================
# FIGURE 1
# =========================================================

def figure1_conceptual_framework(out_dir: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 2.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    xs = [0.08, 0.285, 0.50, 0.715, 0.92]
    labels = [
        "Fixed ML\npipeline $P$",
        "Retrained\nmodel $f_s$",
        "Operational\ndecisions $\\hat{Y}_{s,t}$",
        "Fairness\nviolation $V_{s,t}$",
        "Audit verdict\n$C_{s,t,\\tau}$",
    ]
    subtitles = [
        "",
        "retraining $s$",
        "threshold $t$",
        "measurement",
        r"tolerance $\tau$",
    ]

    for x, label, sublabel in zip(xs, labels, subtitles):
        ax.text(
            x, 0.58, label, ha="center", va="center",
            fontsize=12.5, fontweight="semibold", linespacing=1.12,
            bbox=dict(boxstyle="round,pad=0.34", fc="#F8F9FA", ec="#333333", lw=1.25),
            zorder=3,
        )
        if sublabel:
            ax.text(x, 0.18, sublabel, ha="center", va="center", fontsize=10.5, color="#444444")

    for x1, x2 in zip(xs[:-1], xs[1:]):
        ax.annotate(
            "", xy=(x2 - 0.082, 0.58), xytext=(x1 + 0.082, 0.58),
            arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#333333"), zorder=2,
        )

    ax.set_title(
        "Fairness audit verdicts depend on retraining, operation, and audit tolerance",
        fontsize=14.0, pad=7
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.82, bottom=0.03)
    save_figure(fig, out_dir, "figure1_conceptual_framework", dpi)


# =========================================================
# FIGURE 2
# =========================================================

def figure2_retraining_reproducibility(e1: pd.DataFrame, out_dir: Path, dpi: int):
    summary = (
        e1.groupby(["dataset", "fairness_metric"], as_index=False)["verdict_instability"]
        .mean()
    )

    datasets = ["adult", "acs_income", "compas", "german"]
    metrics = [
        "dpd_violation",
        "dpr_violation",
        "eod_violation",
        "equalized_odds_violation",
    ]

    x = np.arange(len(datasets))
    width = 0.18

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 4.55))

    handles = []
    labels = []
    metric_colors = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["vermillion"]]
    hatches = ["", "//", "..", "xx"]

    for i, metric in enumerate(metrics):
        vals = []
        for d in datasets:
            row = summary[
                (summary["dataset"] == d)
                & (summary["fairness_metric"] == metric)
            ]
            vals.append(row["verdict_instability"].iloc[0] if len(row) else np.nan)

        bars = ax.bar(
            x + (i - 1.5) * width, vals, width=width,
            color=metric_colors[i], edgecolor="#333333", linewidth=0.65, hatch=hatches[i]
        )

        handles.append(bars[0])
        labels.append(label_metric(metric))

    ax.set_xticks(x)
    ax.set_xticklabels([label_dataset(d) for d in datasets])
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Mean verdict instability")
    ymax = summary["verdict_instability"].max()
    ax.set_ylim(0, max(0.35, ymax * 1.10))

    fig.suptitle(
        "Retraining-induced audit-verdict instability",
        fontsize=15.0,
        y=0.985
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.915),
        ncol=4,
        frameon=False,
        columnspacing=1.15,
        handlelength=1.5,
        handletextpad=0.45
    )

    polish_axes(ax, grid=True)
    fig.subplots_adjust(top=0.79, bottom=0.16, left=0.12, right=0.985)

    save_figure(fig, out_dir, "figure2_retraining_reproducibility", dpi)


# =========================================================
# FIGURE 3A
# =========================================================

# def figure3a_metric_vs_verdict(e2: pd.DataFrame, out_dir: Path, dpi: int):
#     fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 4.65))
#
#     ax.scatter(
#         e2["metric_iqr_proxy"],
#         e2["verdict_instability"],
#         alpha=0.24,
#         s=30,
#         color=COLORS["blue"],
#         edgecolors="none"
#     )
#
#     low_high = e2[e2["low_spread_high_instability"] == True].copy()
#     high_zero = e2[e2["high_spread_zero_instability"] == True].copy()
#
#     if len(low_high):
#         ex1 = (
#             low_high
#             .sort_values(["verdict_instability", "metric_iqr_proxy"], ascending=[False, True])
#             .iloc[0]
#         )
#
#         ax.scatter(
#             [ex1["metric_iqr_proxy"]],
#             [ex1["verdict_instability"]],
#             s=125,
#             facecolors="none",
#             edgecolors="black",
#             linewidths=1.8,
#             zorder=5
#         )
#
#         ax.annotate(
#             "Low spread,\nhigh instability",
#             xy=(ex1["metric_iqr_proxy"], ex1["verdict_instability"]),
#             xytext=(35, -50),
#             textcoords="offset points",
#             fontsize=11.5,
#             ha="left",
#             va="top",
#             arrowprops=dict(arrowstyle="->", lw=1.3)
#         )
#
#     if len(high_zero):
#         ex2 = (
#             high_zero
#             .sort_values("metric_iqr_proxy", ascending=False)
#             .iloc[0]
#         )
#
#         ax.scatter(
#             [ex2["metric_iqr_proxy"]],
#             [ex2["verdict_instability"]],
#             s=110,
#             marker="x",
#             linewidths=2.0,
#             zorder=5
#         )
#
#         ax.annotate(
#             "High spread,\nstable verdict",
#             xy=(ex2["metric_iqr_proxy"], ex2["verdict_instability"]),
#             xytext=(-70, 50),
#             textcoords="offset points",
#             fontsize=11.5,
#             ha="right",
#             va="bottom",
#             arrowprops=dict(arrowstyle="->", lw=1.3)
#         )
#
#     ax.set_xlabel(r"Fairness metric 90% range ($Q_{.95}-Q_{.05}$)")
#     ax.set_ylabel(r"Verdict instability $VI$")
#     ax.set_title("Metric spread does not determine verdict stability", pad=10)
#     ax.set_ylim(-0.015, 0.52)
#     polish_axes(ax, grid=True)
#
#     fig.subplots_adjust(
#         top=0.90,
#         bottom=0.15,
#         left=0.13,
#         right=0.98
#     )
#
#     save_figure(fig, out_dir, "figure3a_metric_vs_verdict_stability", dpi)

def figure3a_metric_vs_verdict(e2: pd.DataFrame, out_dir: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 4.65))

    low_high = e2[e2["low_spread_high_instability"] == True].copy()
    high_zero = e2[e2["high_spread_zero_instability"] == True].copy()

    special_mask = (
        e2["low_spread_high_instability"].eq(True)
        | e2["high_spread_zero_instability"].eq(True)
    )
    other = e2[~special_mask].copy()

    # Background observations
    ax.scatter(
        other["metric_iqr_proxy"],
        other["verdict_instability"],
        alpha=0.20,
        s=28,
        color="0.65",
        edgecolors="none",
        zorder=1
    )

    # Low spread / high instability
    ax.scatter(
        low_high["metric_iqr_proxy"],
        low_high["verdict_instability"],
        alpha=0.70,
        s=38,
        color=COLORS["blue"],
        edgecolors="none",
        label="Low spread, high instability",
        zorder=2
    )

    # High spread / stable verdict
    ax.scatter(
        high_zero["metric_iqr_proxy"],
        high_zero["verdict_instability"],
        alpha=0.75,
        s=42,
        color=COLORS["orange"],
        marker="x",
        linewidths=1.5,
        label="High spread, stable verdict",
        zorder=3
    )

    # Representative example: low spread / high instability
    if len(low_high):
        ex1 = (
            low_high
            .sort_values(
                ["verdict_instability", "metric_iqr_proxy"],
                ascending=[False, True]
            )
            .iloc[0]
        )

        ax.scatter(
            [ex1["metric_iqr_proxy"]],
            [ex1["verdict_instability"]],
            s=135,
            facecolors="none",
            edgecolors=COLORS["blue"],
            linewidths=2.2,
            zorder=5
        )

        ax.annotate(
            "Low spread,\nhigh instability",
            xy=(ex1["metric_iqr_proxy"], ex1["verdict_instability"]),
            xytext=(35, -50),
            textcoords="offset points",
            fontsize=11.5,
            ha="left",
            va="top",
            color=COLORS["blue"],
            arrowprops=dict(
                arrowstyle="->",
                lw=1.4,
                color=COLORS["blue"]
            )
        )

    # Representative example: high spread / stable verdict
    if len(high_zero):
        ex2 = (
            high_zero
            .sort_values("metric_iqr_proxy", ascending=False)
            .iloc[0]
        )

        ax.scatter(
            [ex2["metric_iqr_proxy"]],
            [ex2["verdict_instability"]],
            s=125,
            marker="x",
            color=COLORS["orange"],
            linewidths=2.5,
            zorder=5
        )

        ax.annotate(
            "High spread,\nstable verdict",
            xy=(ex2["metric_iqr_proxy"], ex2["verdict_instability"]),
            xytext=(-70, 50),
            textcoords="offset points",
            fontsize=11.5,
            ha="right",
            va="bottom",
            color=COLORS["orange"],
            arrowprops=dict(
                arrowstyle="->",
                lw=1.4,
                color=COLORS["orange"]
            )
        )

    ax.set_xlabel(
        r"Fairness metric 90% range ($Q_{.95}-Q_{.05}$)"
    )
    ax.set_ylabel(r"Verdict instability $VI$")
    ax.set_title(
        "Metric spread does not determine verdict stability",
        pad=10
    )
    ax.set_ylim(-0.015, 0.52)

    polish_axes(ax, grid=True)

    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=10.5
    )

    fig.subplots_adjust(
        top=0.90,
        bottom=0.15,
        left=0.13,
        right=0.98
    )

    save_figure(
        fig,
        out_dir,
        "figure3a_metric_vs_verdict_stability",
        dpi
    )

# =========================================================
# FIGURE 3B
# =========================================================

def figure3b_boundary_vs_verdict(e2: pd.DataFrame, out_dir: Path, dpi: int):
    d = e2.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["standardized_boundary_distance", "verdict_instability"]
    )

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 4.65))

    ax.scatter(
        d["standardized_boundary_distance"],
        d["verdict_instability"],
        alpha=0.24,
        s=30,
        color=COLORS["blue"],
        edgecolors="none"
    )

    ax.set_xlabel(r"Standardized boundary distance  $|\bar{V}-\tau|/SD(V)$")
    ax.set_ylabel(r"Verdict instability $VI$")
    ax.set_title("Boundary proximity is strongly related to verdict instability", pad=10)
    ax.set_ylim(-0.015, 0.52)
    polish_axes(ax, grid=True)

    fig.subplots_adjust(
        top=0.90,
        bottom=0.15,
        left=0.13,
        right=0.98
    )

    save_figure(fig, out_dir, "figure3b_boundary_proximity_vs_instability", dpi)


# =========================================================
# FIGURE 4
# =========================================================

def choose_surface_example(surface: pd.DataFrame):
    preferred = surface[
        (surface["dataset"] == "adult")
        & (surface["model"] == "hist_gradient_boosting")
        & (surface["protected_attribute"] == "sex")
        & (surface["fairness_metric"] == "dpd_violation")
    ]

    if len(preferred):
        return preferred.copy()

    keys = ["dataset", "model", "protected_attribute", "fairness_metric"]
    maxima = (
        surface.groupby(keys)["verdict_instability"]
        .max()
        .sort_values(ascending=False)
    )

    best = maxima.index[0]

    mask = np.ones(len(surface), dtype=bool)
    for k, v in zip(keys, best):
        mask &= surface[k].eq(v)

    return surface.loc[mask].copy()


def figure4_audit_stability_surface(surface: pd.DataFrame, out_dir: Path, dpi: int):
    ex = choose_surface_example(surface)

    pivot = (
        ex.pivot(
            index="audit_tolerance",
            columns="operating_threshold",
            values="pass_probability"
        )
        .sort_index()
        .sort_index(axis=1)
    )

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 4.95))

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
        cmap="viridis",
        interpolation="nearest"
    )

    ax.set_xlabel("Operational threshold $t$")
    ax.set_ylabel("Audit tolerance $\\tau$")

    first = ex.iloc[0]
    title = (
        f"{label_dataset(first['dataset'])} | "
        f"{short_model(first['model'])} | "
        f"{first['protected_attribute']} | "
        f"{label_metric(first['fairness_metric'])}"
    )
    ax.set_title("Representative audit-stability surface\n" + title, pad=10, fontsize=14.0)

    if pivot.columns.min() <= 0.50 <= pivot.columns.max():
        ax.axvline(0.50, linestyle="--", linewidth=1.8, color="white", alpha=0.9)

    cbar = fig.colorbar(im, ax=ax, pad=0.025, fraction=0.055)
    cbar.set_label("Audit pass probability $p(t,\\tau)$")

    save_figure(fig, out_dir, "figure4_audit_stability_surface", dpi)


# =========================================================
# FIGURE 5A
# =========================================================

def figure5a_dataset_model_summary(e4: pd.DataFrame, out_dir: Path, dpi: int):
    """
    One point per dataset-model pair.

    x = SD of accuracy across retrainings
    y = proportion of audit configurations that are unstable

    Color = dataset
    Marker = model
    """
    e4 = e4.copy()
    e4["unstable_flag"] = make_unstable_flag(e4)

    summary = (
        e4.groupby(["dataset", "model"], as_index=False)
        .agg(
            sd_accuracy=("sd_accuracy", "first"),
            unstable_share=("unstable_flag", "mean"),
            mean_vi=("verdict_instability", "mean"),
            max_vi=("verdict_instability", "max"),
        )
    )

    dataset_colors = {
        "adult": COLORS["blue"],
        "acs_income": COLORS["orange"],
        "compas": COLORS["green"],
        "german": COLORS["vermillion"],
    }

    model_markers = {
        "logistic_regression": "o",
        "random_forest": "s",
        "hist_gradient_boosting": "^",
    }

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 5.35))

    for _, r in summary.iterrows():
        ax.scatter(
            r["sd_accuracy"],
            r["unstable_share"],
            s=135,
            color=dataset_colors.get(r["dataset"], "gray"),
            marker=model_markers.get(r["model"], "o"),
            alpha=0.92,
            edgecolor="#222222",
            linewidth=0.65
        )

    ax.set_xlabel("SD of accuracy across retrainings")
    ax.set_ylabel("Share of unstable audit configurations")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Predictive stability does not determine audit stability", pad=10)

    ax.axvline(
        summary["sd_accuracy"].median(),
        linestyle="--",
        linewidth=1.2,
        alpha=0.5,
        color="gray"
    )
    ax.axhline(
        summary["unstable_share"].median(),
        linestyle="--",
        linewidth=1.2,
        alpha=0.5,
        color="gray"
    )

    dataset_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor="#222222",
            markersize=8.5,
            label=label_dataset(ds)
        )
        for ds, color in dataset_colors.items()
    ]

    model_handles = [
        Line2D(
            [0], [0],
            marker=marker,
            linestyle="",
            color="black",
            markerfacecolor="white",
            markeredgewidth=1.2,
            markersize=8.5,
            label=short_model(model)
        )
        for model, marker in model_markers.items()
    ]

    legend1 = fig.legend(
        handles=dataset_handles,
        title="Dataset",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.07),
        ncol=4,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.4
    )
    fig.add_artist(legend1)

    fig.legend(
        handles=model_handles,
        title="Model",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=3,
        frameon=False,
        columnspacing=1.1,
        handletextpad=0.4
    )

    polish_axes(ax, grid=True)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.90, bottom=0.34)

    save_figure(fig, out_dir, "figure5a_predictive_stability_vs_unstable_share", dpi)

    summary.to_csv(
        out_dir / "figure5a_dataset_model_summary.csv",
        index=False
    )


# =========================================================
# FIGURE 5B
# =========================================================

def figure5b_audit_family_support(e4: pd.DataFrame, out_dir: Path, dpi: int):
    """
    One point per audit family.

    x = metric-specific effective subgroup support
    y = proportion of audit tolerances yielding unstable verdicts
    """
    e4 = e4.copy()
    e4["unstable_flag"] = make_unstable_flag(e4)

    if "audit_family" not in e4.columns:
        e4["audit_family"] = (
            e4["dataset"].astype(str) + "__"
            + e4["model"].astype(str) + "__"
            + e4["protected_attribute"].astype(str) + "__"
            + e4["fairness_metric"].astype(str)
        )

    support_col = (
        "effective_support_n"
        if "effective_support_n" in e4.columns
        else "n_min_group"
    )

    fam = (
        e4.groupby(
            [
                "audit_family",
                "dataset",
                "model",
                "protected_attribute",
                "fairness_metric",
            ],
            as_index=False
        )
        .agg(
            effective_support=(support_col, "first"),
            unstable_share=("unstable_flag", "mean"),
            mean_vi=("verdict_instability", "mean"),
            max_vi=("verdict_instability", "max"),
        )
    )

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 4.65))

    ax.scatter(
        fam["effective_support"],
        fam["unstable_share"],
        s=48,
        alpha=0.62,
        color=COLORS["blue"],
        edgecolors="white",
        linewidths=0.35
    )

    ax.set_xscale("log")
    ax.set_xlabel("Metric-specific effective subgroup support")
    ax.set_ylabel("Share of unstable audit tolerances")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Lower subgroup support is associated with broader instability", pad=10)
    polish_axes(ax, grid=True)

    save_figure(fig, out_dir, "figure5b_support_vs_unstable_share", dpi)

    fam.to_csv(
        out_dir / "figure5b_audit_family_summary.csv",
        index=False
    )


# =========================================================
# FIGURE 6
# =========================================================

def figure6_region_composition(surface: pd.DataFrame, out_dir: Path, dpi: int):
    tab = (
        surface.groupby(["dataset", "region"])
        .size()
        .unstack(fill_value=0)
    )

    shares = tab.div(tab.sum(axis=1), axis=0)

    desired_cols = [
        "stable_outside_tolerance",
        "unstable",
        "stable_within_tolerance",
    ]

    for c in desired_cols:
        if c not in shares.columns:
            shares[c] = 0.0

    shares = shares[desired_cols]

    order = [d for d in ["adult", "acs_income", "compas", "german"] if d in shares.index]
    shares = shares.loc[order]

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 4.8))

    bottom = np.zeros(len(shares))

    labels = {
        "stable_outside_tolerance": "Stable outside",
        "unstable": "Unstable",
        "stable_within_tolerance": "Stable within",
    }

    region_colors = {
        "stable_outside_tolerance": COLORS["lightgray"],
        "unstable": COLORS["vermillion"],
        "stable_within_tolerance": COLORS["blue"],
    }

    region_hatches = {
        "stable_outside_tolerance": "//",
        "unstable": "xx",
        "stable_within_tolerance": "",
    }

    x = np.arange(len(shares))

    for col in desired_cols:
        vals = shares[col].to_numpy()
        ax.bar(
            x,
            vals,
            bottom=bottom,
            width=0.68,
            label=labels[col],
            color=region_colors[col],
            hatch=region_hatches[col],
            edgecolor="#333333",
            linewidth=0.75,
        )
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([label_dataset(d) for d in shares.index], fontsize=13)
    ax.set_ylabel("Share of audit surface", fontsize=14)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="y", labelsize=12.5)

    ax.set_title(
        "Stable and unstable audit regions by dataset",
        fontsize=15,
        pad=58,
        weight="semibold",
    )

    legend = ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=3,
        frameon=False,
        fontsize=11.5,
        columnspacing=1.25,
        handlelength=1.8,
        handleheight=1.1,
        handletextpad=0.55,
        borderaxespad=0.0,
    )

    for handle in legend.legend_handles:
        try:
            handle.set_linewidth(0.8)
        except Exception:
            pass

    polish_axes(ax, grid=False)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.20, zorder=0)
    ax.set_axisbelow(True)

    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.15, top=0.72)

    save_figure(fig, out_dir, "figure6_region_composition", dpi)

# =========================================================
# MAIN
# =========================================================

def main():
    args = parse_args()
    setup_style()

    results_dir = Path(args.results)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    e1 = pd.read_csv(require(results_dir / "experiment1_retraining_reproducibility.csv"))
    e2 = pd.read_csv(require(results_dir / "experiment2_metric_vs_verdict_stability.csv"))
    e3 = pd.read_csv(require(results_dir / "experiment3_audit_stability_surface.csv"))
    e4 = pd.read_csv(require(results_dir / "experiment4_predictive_vs_audit_stability.csv"))

    figure1_conceptual_framework(out_dir, args.dpi)
    figure2_retraining_reproducibility(e1, out_dir, args.dpi)
    figure3a_metric_vs_verdict(e2, out_dir, args.dpi)
    figure3b_boundary_vs_verdict(e2, out_dir, args.dpi)
    figure4_audit_stability_surface(e3, out_dir, args.dpi)
    figure5a_dataset_model_summary(e4, out_dir, args.dpi)
    figure5b_audit_family_support(e4, out_dir, args.dpi)
    figure6_region_composition(e3, out_dir, args.dpi)

    print("\nFinal figure generation complete.")
    print(f"Output directory: {out_dir.resolve()}")
    print("\nGenerated files:")
    for p in sorted(out_dir.iterdir()):
        if p.suffix.lower() in {".png", ".pdf", ".csv"}:
            print(" ", p.name)


if __name__ == "__main__":
    main()
