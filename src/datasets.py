from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
import numpy as np
import pandas as pd
import requests

@dataclass
class DatasetBundle:
    name: str
    X: pd.DataFrame
    y: pd.Series
    protected: dict[str, pd.Series]
    numeric_cols: list[str]
    categorical_cols: list[str]

def _clean_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    return df

def load_adult():
    import openml
    ds = openml.datasets.get_dataset(1590)
    X, y, _, _ = ds.get_data(target=ds.default_target_attribute)
    X = _clean_columns(X)
    y = pd.Series(y).astype(str).str.contains(">50K").astype(int)

    # Protected attributes are retained in X for prediction because that reflects
    # common benchmark usage. They are also exposed separately for auditing.
    sex_col = next(c for c in X.columns if c.lower() == "sex")
    race_col = next(c for c in X.columns if c.lower() == "race")

    protected = {
        "sex": (X[sex_col].astype(str).str.lower() != "male").astype(int),
        "race": (X[race_col].astype(str).str.lower() != "white").astype(int),
    }

    cat = [c for c in X.columns if str(X[c].dtype) in ("category", "object", "string", "bool")]
    num = [c for c in X.columns if c not in cat]
    return DatasetBundle("adult", X.reset_index(drop=True), y.reset_index(drop=True),
                         {k:v.reset_index(drop=True) for k,v in protected.items()}, num, cat)

def load_german():
    import openml
    ds = openml.datasets.get_dataset(31)
    X, y, _, _ = ds.get_data(target=ds.default_target_attribute)
    X = _clean_columns(X)
    y = pd.Series(y).astype(str).str.lower().isin(["good", "1", "true"]).astype(int)

    # Common credit-g coding:
    # personal_status includes sex information.
    ps = next((c for c in X.columns if c.lower() in ("personal_status", "personal_status_and_sex")), None)
    age = next((c for c in X.columns if c.lower() == "age"), None)
    protected = {}

    if ps is not None:
        # OpenML credit-g commonly exposes decoded labels such as
        # "male single", "male div/sep", "female div/dep/mar", etc.
        # Encode 0 = male, 1 = female for the audit comparison.
        status = X[ps].astype(str).str.strip().str.lower()
        female_mask = status.str.contains("female", na=False)
        male_mask = status.str.contains("male", na=False)

        # Defensive fallback if raw Statlog codes are returned.
        raw = X[ps].astype(str).str.strip().str.upper()
        female_mask = female_mask | raw.isin({"A92", "A95"})
        male_mask = male_mask | raw.isin({"A91", "A93", "A94"})

        unknown = ~(female_mask | male_mask)
        if unknown.any():
            bad = sorted(X.loc[unknown, ps].astype(str).unique().tolist())
            raise ValueError(
                f"Unrecognized German Credit personal-status values for sex decoding: {bad}"
            )

        protected["sex"] = female_mask.astype(int)
    if age is not None:
        # Common benchmark convention: age < 25 as the comparison group.
        protected["age_lt_25"] = (pd.to_numeric(X[age], errors="coerce") < 25).astype(int)

    cat = [c for c in X.columns if str(X[c].dtype) in ("category", "object", "string", "bool")]
    num = [c for c in X.columns if c not in cat]
    return DatasetBundle("german", X.reset_index(drop=True), y.reset_index(drop=True),
                         {k:v.reset_index(drop=True) for k,v in protected.items()}, num, cat)

def load_compas():
    url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))

    # Standard ProPublica-style filtering.
    df = df[
        (df["days_b_screening_arrest"] <= 30) &
        (df["days_b_screening_arrest"] >= -30) &
        (df["is_recid"] != -1) &
        (df["c_charge_degree"] != "O") &
        (df["score_text"] != "N/A")
    ].copy()

    # Predict two-year recidivism.
    y = df["two_year_recid"].astype(int)

    # Use common pre-decision features only.
    features = [
        "age", "sex", "race", "juv_fel_count", "juv_misd_count",
        "juv_other_count", "priors_count", "c_charge_degree"
    ]
    X = df[features].copy()
    protected = {
        "sex": (df["sex"].astype(str).str.lower() != "male").astype(int),
        # Binary Black-vs-White analysis; other groups dropped below.
    }

    bw = df["race"].isin(["African-American", "Caucasian"])
    X = X.loc[bw].reset_index(drop=True)
    y = y.loc[bw].reset_index(drop=True)
    protected = {
        "sex": protected["sex"].loc[bw].reset_index(drop=True),
        "race": (df.loc[bw, "race"] == "African-American").astype(int).reset_index(drop=True),
    }

    cat = ["sex", "race", "c_charge_degree"]
    num = [c for c in X.columns if c not in cat]
    return DatasetBundle("compas", X, y, protected, num, cat)

def load_acs_income(state="CA", year=2018, horizon="1-Year"):
    from folktables import ACSDataSource, ACSIncome

    source = ACSDataSource(survey_year=str(year), horizon=horizon, survey="person")
    data = source.get_data(states=[state], download=True)
    X_np, y_np, group_np = ACSIncome.df_to_numpy(data)

    cols = list(ACSIncome.features)
    X = pd.DataFrame(X_np, columns=cols)
    y = pd.Series(y_np.astype(int))

    # Folktables ACSIncome group is RAC1P. We also derive sex from SEX.
    protected = {}
    if "SEX" in X.columns:
        # ACS PUMS: SEX 1=Male, 2=Female.
        protected["sex"] = (X["SEX"].astype(float) != 1.0).astype(int)
    if "RAC1P" in X.columns:
        # Common binary benchmark: White alone (1) vs non-White.
        protected["race"] = (X["RAC1P"].astype(float) != 1.0).astype(int)

    # Folktables features are numeric codes; categorical-code columns are treated categorically.
    categorical_candidates = {
        "COW","SCHL","MAR","OCCP","POBP","RELP","WKHP","SEX","RAC1P"
    }
    cat = [c for c in X.columns if c in categorical_candidates]
    num = [c for c in X.columns if c not in cat]
    for c in cat:
        X[c] = X[c].astype("Int64").astype(str)

    return DatasetBundle("acs_income", X.reset_index(drop=True), y.reset_index(drop=True),
                         {k:v.reset_index(drop=True) for k,v in protected.items()}, num, cat)

def _cap_rows(bundle, max_rows, seed=2027):
    if max_rows is None or len(bundle.y) <= int(max_rows):
        return bundle
    max_rows = int(max_rows)
    rng = np.random.default_rng(seed)

    # Stratify approximately by target using proportional sampling.
    idx_parts = []
    y = bundle.y.reset_index(drop=True)
    for label, idx in y.groupby(y).groups.items():
        idx = np.asarray(list(idx))
        take = max(1, int(round(max_rows * len(idx) / len(y))))
        take = min(take, len(idx))
        idx_parts.append(rng.choice(idx, size=take, replace=False))
    idx = np.concatenate(idx_parts)
    if len(idx) > max_rows:
        idx = rng.choice(idx, size=max_rows, replace=False)
    elif len(idx) < max_rows:
        remaining = np.setdiff1d(np.arange(len(y)), idx, assume_unique=False)
        add = rng.choice(remaining, size=min(max_rows-len(idx), len(remaining)), replace=False)
        idx = np.concatenate([idx, add])

    idx = np.sort(idx)
    bundle.X = bundle.X.iloc[idx].reset_index(drop=True)
    bundle.y = bundle.y.iloc[idx].reset_index(drop=True)
    bundle.protected = {k: v.iloc[idx].reset_index(drop=True) for k,v in bundle.protected.items()}
    return bundle

def load_dataset(name, cfg):
    if name == "adult":
        bundle = load_adult()
    elif name == "compas":
        bundle = load_compas()
    elif name == "german":
        bundle = load_german()
    elif name == "acs_income":
        bundle = load_acs_income(
            state=cfg.get("state", "CA"),
            year=cfg.get("year", 2018),
            horizon=cfg.get("horizon", "1-Year"),
        )
    else:
        raise ValueError(name)

    return _cap_rows(bundle, cfg.get("max_rows"))
