from pathlib import Path
import yaml
from src.metrics import fairness_metrics

cfg = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
assert cfg["split"]["train"] + cfg["split"]["validation"] + cfg["split"]["test"] == 1.0

m = fairness_metrics(
    y_true=[1,1,0,0,1,0,1,0],
    y_pred=[1,0,0,0,1,1,0,0],
    protected=[0,0,0,0,1,1,1,1],
)
assert "dpd_violation" in m
assert 0 <= m["dpd_violation"] <= 1
assert 0 <= m["dpr_violation"] <= 1
print("Basic validation passed.")
