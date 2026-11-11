# Colab: Retrain Module 2 on track features (for you to run)

Live ops still use **rules_v1**. After you train a clean track-feature model, place it at:

`models/module2/baselines/xgboost_track_features.joblib`

and we can wire it behind the existing sudden-stop gates (AND logic).

## Why not the old joblib?

Old Module 2 tables leak via `has_any_anomaly_in_video` and use whole-frame motion — not aligned with live ByteTrack features.

## Steps

1. On your PC (backend venv):

```bash
cd TrafficAI_ControlCenter
app/backend/.venv/Scripts/python.exe scripts/export_track_features.py
```

2. Upload `data/exports/track_features_module2.csv` to Colab Drive.

3. Colab skeleton:

```python
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, precision_recall_fscore_support
from xgboost import XGBClassifier
import joblib

df = pd.read_csv("track_features_module2.csv")
# Baseline label = sudden_stop_candidate (RULE). Replace with human labels when available.
y = df["sudden_stop_candidate"].astype(int)
X = df[[
  "frames","stationary_ratio","max_speed","last_speed","mean_recent_speed",
  "stationary_duration_sec","moving_frames","was_moving","currently_stopped",
  "distance_proxy_m","uturn_score"
]]
groups = df["video"]

# Always report rule baseline (= y itself here if y is rule-derived)
print("rule-as-label sanity: if y is sudden_stop_candidate, ML F1≈1 means you only relearned the rule")

gkf = GroupKFold(n_splits=min(5, groups.nunique()))
f1s = []
for tr, te in gkf.split(X, y, groups):
    clf = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss",
    )
    clf.fit(X.iloc[tr], y.iloc[tr])
    pred = clf.predict(X.iloc[te])
    f1s.append(f1_score(y.iloc[te], pred, zero_division=0))
print("mean F1", sum(f1s)/len(f1s))

joblib.dump({"model": clf, "features": list(X.columns)}, "xgboost_track_features.joblib")
```

4. **Acceptance before wiring into API**

- Drop any video-level leak columns (none in the export by design).
- Prefer independent labels for true incidents.
- Publish **rule-only** precision/recall next to model metrics.
- Group split by `video`.

5. Send the joblib back into `models/module2/baselines/` and ask to wire hybrid inference.
