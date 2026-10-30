import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

EPS = 1e-12

def _rate(mask, values):
    n = int(mask.sum())
    if n == 0:
        return np.nan, 0
    return float(np.mean(values[mask])), n

def predictive_metrics(y_true, prob, y_pred):
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        out["auroc"] = float(roc_auc_score(y_true, prob))
    except ValueError:
        out["auroc"] = np.nan
    return out

def fairness_metrics(y_true, y_pred, protected,
                     min_group_n=20,
                     min_positive_group_n=10,
                     min_negative_group_n=10):
    """Compute fairness metrics and support/validity flags.

    protected: 0 = reference group, 1 = comparison group.
    Definedness and analysis support are deliberately separated.
    """
    y_true=np.asarray(y_true).astype(int)
    y_pred=np.asarray(y_pred).astype(int)
    a=np.asarray(protected).astype(int)

    g0=a==0; g1=a==1
    sr0,n0=_rate(g0,y_pred); sr1,n1=_rate(g1,y_pred)
    pos0=g0&(y_true==1); pos1=g1&(y_true==1)
    neg0=g0&(y_true==0); neg1=g1&(y_true==0)
    tpr0,n_pos0=_rate(pos0,y_pred); tpr1,n_pos1=_rate(pos1,y_pred)
    fpr0,n_neg0=_rate(neg0,y_pred); fpr1,n_neg1=_rate(neg1,y_pred)

    dpd_defined=np.isfinite(sr0) and np.isfinite(sr1)
    dpr_defined=dpd_defined and max(sr0,sr1)>EPS
    eod_defined=np.isfinite(tpr0) and np.isfinite(tpr1)
    eqodds_defined=eod_defined and np.isfinite(fpr0) and np.isfinite(fpr1)

    dpd=(sr1-sr0) if dpd_defined else np.nan
    eod=(tpr1-tpr0) if eod_defined else np.nan
    eqodds=max(abs(tpr1-tpr0),abs(fpr1-fpr0)) if eqodds_defined else np.nan
    dpr_sym=min(sr0,sr1)/max(sr0,sr1) if dpr_defined else np.nan

    min_group=min(n0,n1); min_pos=min(n_pos0,n_pos1); min_neg=min(n_neg0,n_neg1)
    group_ok=min_group>=int(min_group_n)
    pos_ok=min_pos>=int(min_positive_group_n)
    neg_ok=min_neg>=int(min_negative_group_n)

    return {
        "n_group0":n0,"n_group1":n1,"n_min_group":min_group,
        "n_positive_group0":n_pos0,"n_positive_group1":n_pos1,"n_min_positive_group":min_pos,
        "n_negative_group0":n_neg0,"n_negative_group1":n_neg1,"n_min_negative_group":min_neg,
        "selection_rate_group0":sr0,"selection_rate_group1":sr1,
        "tpr_group0":tpr0,"tpr_group1":tpr1,"fpr_group0":fpr0,"fpr_group1":fpr1,
        "dpd_signed":dpd,"dpd_violation":abs(dpd) if dpd_defined else np.nan,
        "dpr_symmetric":dpr_sym,"dpr_violation":1.0-dpr_sym if dpr_defined else np.nan,
        "eod_signed":eod,"eod_violation":abs(eod) if eod_defined else np.nan,
        "equalized_odds_violation":eqodds,
        "dpd_defined":bool(dpd_defined),"dpr_defined":bool(dpr_defined),
        "eod_defined":bool(eod_defined),"equalized_odds_defined":bool(eqodds_defined),
        "min_group_ok":bool(group_ok),"min_positive_group_ok":bool(pos_ok),"min_negative_group_ok":bool(neg_ok),
        "dpd_analysis_valid":bool(dpd_defined and group_ok),
        "dpr_analysis_valid":bool(dpr_defined and group_ok),
        "eod_analysis_valid":bool(eod_defined and group_ok and pos_ok),
        "equalized_odds_analysis_valid":bool(eqodds_defined and group_ok and pos_ok and neg_ok),
    }

FAIRNESS_VIOLATION_COLUMNS=["dpd_violation","dpr_violation","eod_violation","equalized_odds_violation"]
FAIRNESS_VALIDITY_COLUMNS={
    "dpd_violation":"dpd_analysis_valid",
    "dpr_violation":"dpr_analysis_valid",
    "eod_violation":"eod_analysis_valid",
    "equalized_odds_violation":"equalized_odds_analysis_valid",
}
