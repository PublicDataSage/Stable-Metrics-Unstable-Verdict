# Stable Metrics, Unstable Verdicts: Reproducibility in Algorithmic Fairness Auditing

This repository contains the experimental code, analysis scripts, and selected results for **“Stable Metrics, Unstable Verdicts: Reproducibility in Algorithmic Fairness Auditing.”**

Fairness audits often compare a quantitative group-disparity measure with a specified tolerance and report a categorical conclusion. A single conclusion does not show whether the same result would be obtained after another realization of the modeling pipeline or under a different evaluation procedure. This project evaluates the reproducibility of those conclusions across model realizations, operational decision thresholds, audit tolerances, and two evaluation protocols.

The study concerns **reproducibility relative to specified audit criteria**, not whether any particular fairness criterion, tolerance, or decision system is normatively appropriate.

## Evaluation protocols

The experiments use two distinct procedures, each with 100 model realizations for every dataset–model combination.

| Protocol | Training and validation | Evaluation cohort | Main purpose |
|---|---|---|---|
| **Varying evaluation** (`pi_vary`) | Regenerated across seeds | Test partition also changes across seeds | Characterize verdict variation when modeling and evaluation partitions vary |
| **Fixed evaluation** (`pi_fixed`) | Regenerated across seeds from the remaining development data | One common 15% held-out test partition per dataset, selected with seed `2027` | Characterize verdict variation when independently fitted models share an evaluation cohort |

Both protocols use overall **70% training / 15% validation / 15% test** proportions. Under fixed evaluation, the same test examples are reused across seeds and model families; a per-seed split of the remaining 85% creates the training and validation partitions. The validation partitions are not used for hyperparameter selection in the reported experiments. Model settings are fixed in advance.

**Interpretation:** Differences between these protocols are descriptive. Holding the evaluation cohort fixed does not isolate a causal effect of evaluation sampling because the resulting training-data distributions and fitted models can also differ between protocols.

## Audit-verdict formulation

Let `P` denote a modeling pipeline, `pi` an evaluation protocol, `s` a realization, `t` an operational decision threshold, and `tau` an audit tolerance. For a nonnegative fairness-violation statistic `V`, the categorical audit verdict is

\[
C(P;\pi,s,t,\tau)=\mathbb{I}[V_{s,t}\leq\tau],
\]

provided that the audit measurement is defined and satisfies its metric-specific support requirements. The verdict is otherwise undefined. The value `1` means **within the specified tolerance**, not that the model is universally fair.

Let `R_{s,t}` indicate measurement validity. Conditional on positive validity probability, the within-tolerance probability is

\[
p_{P,\pi}(t,\tau)=\Pr_\pi\!\left(V_{S,t}\leq\tau\mid R_{S,t}=1\right).
\]

For two independent draws from the same validity-conditioned realization distribution, pairwise verdict instability and audit agreement are

\[
VI_{P,\pi}(t,\tau)=2p_{P,\pi}(t,\tau)\bigl[1-p_{P,\pi}(t,\tau)\bigr],
\qquad AA_{P,\pi}(t,\tau)=1-VI_{P,\pi}(t,\tau).
\]

For fixed evaluation, these probabilities are conditional on the particular common evaluation cohort. `VI` ranges from `0` (no pairwise disagreement) to `0.5` (maximal disagreement). These are standard Bernoulli probability relationships; the study evaluates their behavior under the specified audit procedures.

The **Audit Stability Surface** is `A_{P,pi}(t,tau) = p_{P,pi}(t,tau)`. With stability parameter `epsilon = 0.10`, valid surface points are classified as:

- **Stable outside tolerance:** `p <= epsilon`.
- **Unstable:** `epsilon < p < 1 - epsilon`.
- **Stable within tolerance:** `p >= 1 - epsilon`.

We also examine `epsilon = 0.05` and `0.20`. These regions describe reproducibility, not the substantive desirability of a fairness result.

## Datasets, models, and audit settings

| Component | Specification |
|---|---|
| Datasets | Adult, ACSIncome, COMPAS, German Credit (`credit-g`) |
| Model families | Logistic Regression (LR), Random Forest (RF), HistGradientBoosting (HGB) |
| Realizations | 100 seeds (`0`–`99`) per dataset–model pair, per protocol |
| Fitted models | 1,200 for varying evaluation; 1,200 for fixed evaluation |
| Operational thresholds | `0.30` through `0.70`, step `0.05` |
| Reference operational threshold | `t = 0.50` |
| Fairness criteria | Demographic parity difference (DPD), demographic parity ratio (DPR), equal opportunity difference (EOD), equalized odds (EO) |
| Audit tolerances | DPD/EOD/EO: `0.00`–`0.30` by `0.01`; DPR: ratio `0.60`–`1.00` by `0.01`, converted to violation tolerance `1 - ratio` |
| Primary stability parameter | `epsilon = 0.10` |
| Fixed-test partition seed | `2027` |

**Data sources:** Adult and German Credit are obtained from OpenML; COMPAS uses publicly available ProPublica data; ACSIncome uses the American Community Survey via Folktables. The reported ACSIncome sample uses California's 2018 one-year ACS data, with a deterministic, approximately target-stratified cap of 20,000 rows configured in `config.yaml` (`max_rows: null` removes the cap).

All criteria are represented as nonnegative violation statistics. For the symmetric demographic parity ratio, the violation is `1 - DPR`, allowing a consistent within-tolerance rule. The German Credit loader supports decoded OpenML sex labels and raw Statlog personal-status codes.

### Support and validity requirements

A measurement is included only if its metric-specific requirements are satisfied:

| Criterion | Required evaluation support |
|---|---|
| DPD | At least 20 observations in each protected group |
| DPR | Same as DPD, plus a defined selection-rate denominator |
| EOD | At least 20 observations per group and 10 positive-label observations per group |
| EO | At least 20 observations per group, 10 positive-label and 10 negative-label observations per group |

Undefined or insufficiently supported measurements remain marked as invalid; they are **not** counted as within or outside tolerance. Configurations with no valid realizations are excluded from stability estimates. Effective subgroup support is based on the counts required for each fairness criterion.

## Analyses

1. **Verdict reproducibility:** Estimates within-tolerance probability, pairwise instability, and stability-region membership at the reference operational threshold.
2. **Metric versus verdict stability:** Compares continuous fairness-metric spread with categorical disagreement, including descriptive associations with standardized audit-boundary distance.
3. **Joint audit stability:** Evaluates the Audit Stability Surface over the operational-threshold and audit-tolerance grids, including matched comparisons between evaluation protocols.
4. **Statistical associations:** Examines effective subgroup support, variability in predictive accuracy, and boundary distance using audit-family-clustered OLS and supplementary fractional GEE specifications. The fixed-evaluation regression analysis uses 3,123 matched reference-threshold configurations from 93 audit families.

The robustness workflow also evaluates matched audit-family differences, dataset–model-block bootstrap sensitivity, alternative stability parameters, finer audit-tolerance grids, pointwise Wilson uncertainty, alternative dependence assumptions, and leave-one-dataset-out regression checks.

## Repository layout

The layout below identifies the principal scripts and output groups. The exact set of generated intermediate files depends on which commands have been run. Files in `results/` are **outputs**, not additional source-code dependencies.

```text
fairness_audit_stability/
├── README.md
├── .gitignore
├── requirements.txt
├── config.yaml
├── src/
│   ├── __init__.py
│   ├── datasets.py
│   ├── models.py
│   └── metrics.py
├── validate_setup.py
├── diagnose_quick_test.py
├── run_experiments.py
├── analyze_experiments.py
├── run_fixed_test_robustness.py
├── validate_statistical_results.py
├── generate_figures.py
├── generate_robustness_figures.py
├── results/
│   ├── threshold_metrics.csv
│   ├── run_manifest.csv
│   ├── metadata.json
│   ├── experiment1_retraining_reproducibility.csv
│   ├── experiment2_metric_vs_verdict_stability.csv
│   ├── experiment3_audit_stability_surface.csv
│   ├── experiment4_predictive_vs_audit_stability.csv
│   ├── ... other original analysis outputs ...
│   ├── fixed_test_robustness/
│   │   ├── run_protocol.json
│   │   ├── metadata.json
│   │   ├── config_fixed_test.yaml
│   │   ├── run_manifest.csv
│   │   ├── threshold_metrics.csv
│   │   ├── fixed_test_indices_*.csv
│   │   ├── fixed_test_indices_*.json
│   │   ├── experiment1_retraining_reproducibility.csv
│   │   ├── experiment3_audit_stability_surface.csv
│   │   ├── experiment4_predictive_vs_audit_stability.csv
│   │   ├── ... other fixed-evaluation analysis outputs ...
│   │   └── checkpoints/                   # Generated; normally untracked
│   └── statistical_validation/
│       ├── validation_checks.csv
│       ├── validation_manifest.json
│       ├── matched_coverage.csv
│       ├── unmatched_surface_keys.csv
│       ├── matched_surface_points.csv
│       ├── matched_summary.csv
│       ├── matched_audit_family_effects.csv
│       ├── matched_dataset_model_block_bootstrap.csv
│       ├── epsilon_and_uncertainty_summary.csv
│       ├── uncertainty_by_surface_point.csv
│       ├── regression_coefficients_and_standard_errors.csv
│       ├── regression_fit_status.csv
│       ├── regression_numerical_diagnostics.csv
│       ├── saved_vs_recomputed_regressions.csv
│       └── leave_one_dataset_out_ols.csv
└── figures/
    ├── figure3a_metric_vs_verdict_stability.pdf
    ├── figure8_matched_protocol_unstable_share.pdf
    ├── figure9_matched_audit_family_protocol_shift.pdf
    ├── figure11_uncertainty_region_composition.pdf
    ├── figure12_paired_protocol_stability_surface.pdf
    └── ... other generated figures ...
```

## Installation

Clone the repository:

```bash
git clone https://github.com/PublicDataSage/fairness_audit_stability.git
cd fairness_audit_stability
```

Install dependencies using an available Python environment. **A virtual environment is not required.** On Windows PowerShell:

```powershell
python -m pip install -r requirements.txt
```

The statistical-validation script additionally uses `scipy` and `statsmodels`; confirm they are installed if they are not already specified in your dependency file:

```powershell
python -m pip install scipy statsmodels
```

On Linux or macOS, replace `python` with `python3` when needed. Use a consistent Python interpreter and package versions for reproducibility.

## Reproduce the analyses

Run the commands below from the repository root. **Keep original and fixed-evaluation results in separate directories.** The commands assume all four datasets and all three models are enabled in `config.yaml`.

### 1. Verify the environment

```powershell
python validate_setup.py
```

An optional small-run check is:

```powershell
python run_experiments.py --config config.yaml --seeds 3 --datasets adult german
python diagnose_quick_test.py
```

This smoke test writes to the configured output directory. For the final analysis, use a clean output directory or confirm that incomplete smoke-test outputs will not be mixed with full-run results.

### 2. Run and analyze varying-evaluation experiments

```powershell
python run_experiments.py --config config.yaml
python analyze_experiments.py --config config.yaml
```

This produces the original `results/threshold_metrics.csv` and Experiment 1–4 analysis files. The configured `output_dir` should point to `results`.

### 3. Run and analyze fixed-evaluation experiments

```powershell
python run_fixed_test_robustness.py --config config.yaml --out results/fixed_test_robustness --seeds 100 --fixed-test-seed 2027
python analyze_experiments.py --config results/fixed_test_robustness/config_fixed_test.yaml
```

The fixed-test runner creates per-seed checkpoints for resumption, verifies the fixed partition against a stored dataset fingerprint, and writes a run manifest. Its generated `config_fixed_test.yaml` points the analysis script to the fixed-evaluation outputs. When running on another machine, regenerate this file or check its `output_dir`, which may contain a machine-specific absolute path.

### 4. Validate and compare protocols

```powershell
python validate_statistical_results.py --original-dir results --fixed-dir results/fixed_test_robustness --out results/statistical_validation --primary-t 0.50 --epsilon 0.10 --bootstrap 1000 --seed 2027 --require-original
```

This script **does not retrain models or replace the experiment results**. It checks result integrity, constructs matched configuration comparisons, estimates Wilson intervals and stability-region uncertainty, produces descriptive dataset–model-block bootstrap summaries, and refits statistical models for sensitivity analysis.

Inspect `results/statistical_validation/validation_checks.csv` and `validation_manifest.json` for failed checks. Also inspect `regression_fit_status.csv` and `regression_numerical_diagnostics.csv`: passing file-integrity checks does not imply that every statistical specification converged reliably. The binary unstable-region GEE has shown numerical instability and is not used as primary evidence.

### 5. Generate figures

```powershell
python generate_figures.py --results results --out figures --dpi 300
python generate_robustness_figures.py --validation results/statistical_validation --out figures --dpi 300
```

`generate_figures.py` generates the original-analysis figures. `generate_robustness_figures.py` generates Figures 7–12 from the **validated comparison CSVs**, exporting vector PDF and PNG versions. The latter requires `matched_surface_points.csv`, `matched_audit_family_effects.csv`, `epsilon_and_uncertainty_summary.csv`, and `uncertainty_by_surface_point.csv` in the validation directory; it does not require the large raw `audit_long.csv`.

## Figures used in the manuscript

The manuscript's figure numbers differ from the generation-script filename prefixes. These are the five PDF figures used in the current draft:

| Manuscript | Committed/generated filename | Contents |
|---|---|---|
| Figure 1 | `figure3a_metric_vs_verdict_stability.pdf` | Continuous metric spread versus categorical verdict instability |
| Figure 2 | `figure12_paired_protocol_stability_surface.pdf` | Paired stability surfaces for varying and fixed evaluation |
| Figure 3 | `figure8_matched_protocol_unstable_share.pdf` | Dataset-level unstable-region prevalence under both protocols |
| Figure 4 | `figure9_matched_audit_family_protocol_shift.pdf` | Mean instability across 93 matched audit families |
| Figure 5 | `figure11_uncertainty_region_composition.pdf` | Reliability of region classifications under pointwise Wilson intervals |

Other generated PDFs may be retained for exploratory analysis or a supplementary archive but are not required for the current manuscript figures.

## Reproducing published numerical results

The main matched comparison includes **28,107 configurations across 93 audit families**. In the reported analysis, mean pairwise verdict instability was `0.1239` under varying evaluation and `0.0690` under fixed evaluation; the corresponding unstable-region shares were `28.33%` and `15.52%` at `epsilon = 0.10`. Disagreement persisted even when independently fitted models were evaluated on a shared test cohort. These are descriptive results for the specified datasets, thresholds, and protocol implementations.

Three German Credit age-under-25 equalized-odds audit families did not satisfy the fixed-test support requirements; unmatched configurations are not included in the paired comparison. The matching inventory is recorded in `matched_coverage.csv` and `unmatched_surface_keys.csv`.

The following files support different levels of reproduction:

| Goal | Files needed |
|---|---|
| Inspect headline results | `matched_summary.csv`, `epsilon_and_uncertainty_summary.csv`, `validation_checks.csv` |
| Regenerate the four robustness figures in the manuscript | `matched_surface_points.csv`, `matched_audit_family_effects.csv`, `epsilon_and_uncertainty_summary.csv`, `uncertainty_by_surface_point.csv` |
| Rerun statistical validation without retraining | Both protocols' `threshold_metrics.csv`, Experiment 1 and 3 outputs; fixed-evaluation Experiment 4 output; original and fixed metadata/manifest where available |
| Reproduce everything from source | Source scripts, `config.yaml`, dataset definitions and original sources, 100 seeds per model, and the commands above |

## What to commit to Git

**Commit:** source code, `config.yaml`, dependency specifications, README, the five manuscript PDFs, and a compact but sufficient set of processed results. To regenerate the final plots without retraining, retain the four validation inputs identified in the table above. If repository size permits, also retain both protocols' Experiment 1 and Experiment 3 summaries, fixed-evaluation Experiment 4 summaries, regression diagnostics, and run metadata.

**Generally omit:** downloaded raw datasets, per-seed checkpoint folders, Python caches, virtual environments, temporary files, redundant PNG duplicates, and oversized intermediate tables such as `audit_long.csv`. Do not commit generated files that disclose machine-specific paths without reviewing them. If full intermediate data are needed for archival reproducibility, deposit them in a versioned external archive and reference that archive here.

Example `.gitignore` entries:

```gitignore
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
venv/
results/fixed_test_robustness/checkpoints/
data/
*.tmp
```

Avoid adding a blanket `results/` ignore rule if you intend to version processed results. GitHub's regular file limit is 100 MB; very large files require an alternate storage strategy such as Git LFS or a data archive.

## Statistical interpretation and limitations

- The two audit protocols describe different joint distributions of modeling and evaluation realizations. Their contrast does not identify the isolated causal effect of evaluation-cohort sampling.
- Wilson intervals are **pointwise** and rely on an independent-realization approximation; they are not simultaneous confidence bands over the full surface.
- Audit-tolerance-expanded configurations reuse underlying models and are dependent. Regression specifications account for some dependence but do not identify causal effects.
- Bootstrap summaries use 12 dataset–model blocks and are treated as descriptive sensitivity intervals, not population-level causal confidence intervals.
- Standardized boundary distance and verdict instability both depend on the fairness-violation distribution. Their statistical association is descriptive and should not be interpreted as an independent causal mechanism.
- The results apply to the analyzed benchmarks, modeling configurations, evaluation cohorts, fairness criteria, and audit grids. Stable verdicts do not establish that the selected fairness criterion or tolerance is substantively appropriate.

## Citation and availability

Manuscript: **“Stable Metrics, Unstable Verdicts: Reproducibility in Algorithmic Fairness Auditing.”** Please add final publication metadata, DOI, or archival preprint details here when available. The original datasets are accessible from their respective public providers; the project provides processing and analysis code rather than redistributing those datasets.
