# Artifact Naming Convention (Phase A Freeze)

**Status:** FROZEN for Agents B/C/E  
**Do not invent alternate names.**

## Model roster (11 per module)

### Base ML (4)
| `model_key` | Family |
|---|---|
| `logistic_regression` | sklearn |
| `random_forest` | sklearn |
| `xgboost` | XGBoost |
| `lightgbm` | LightGBM |

### Optimized DL (7)
| `model_key` | Family |
|---|---|
| `mlp` | PyTorch tabular MLP |
| `lstm` | PyTorch sequence |
| `gru` | PyTorch sequence |
| `bilstm` | PyTorch sequence |
| `tcn` | PyTorch temporal CNN |
| `transformer_encoder` | PyTorch |
| `temporal_cnn` | PyTorch 1D-CNN |

**Removed from arena:** `gradient_boosting`, `gat_gru`, `decision_policy_head`, dual near-identical transformers.

## Directory layout

```text
models/
  module{N}/
    baselines/
      {model_key}__{schema_id}__{train_date}__v{ver}.joblib
      {model_key}__{schema_id}__{train_date}__v{ver}.meta.json
    dl/
      {model_key}__{schema_id}__{train_date}__v{ver}.pt
      {model_key}__{schema_id}__{train_date}__v{ver}.meta.json
      {model_key}__{schema_id}__{train_date}__v{ver}.norm.npz   # mean/std if needed
    deployment/
      winner.json
    schema_hash.txt
```

## Filename pattern

```text
{model_key}__{schema_id}__{YYYYMMDD}__v{MAJOR.MINOR}.{ext}
```

Examples:
- `xgboost__module2_breakdown_v1__20261112__v1.0.joblib`
- `lstm__module2_breakdown_v1__20261112__v1.0.pt`
- `lightgbm__module1_bus_priority_v1__20261112__v1.0.joblib`

## Required `.meta.json` fields

```json
{
  "model_key": "xgboost",
  "module": "module2",
  "schema_id": "module2_breakdown_v1",
  "schema_version": "1.0.0",
  "train_date": "20261112",
  "artifact_version": "1.0",
  "feature_names": ["frames", "stationary_ratio"],
  "label_name": "y_abnormal_stop",
  "metrics_val": {"f1": 0.0, "pr_auc": 0.0},
  "metrics_test": {"f1": 0.0, "pr_auc": 0.0},
  "latency_ms_p50": 0.0,
  "selected_as_winner": false,
  "framework": "xgboost|sklearn|pytorch",
  "notes": ""
}
```

## Deployment winner file

`models/module{N}/deployment/winner.json`:

```json
{
  "module": "module2",
  "model_key": "xgboost",
  "artifact_relpath": "baselines/xgboost__module2_breakdown_v1__20261112__v1.0.joblib",
  "selection_rule": "best_validation_f1_then_pr_auc_then_lower_latency",
  "selected_on": "validation_only",
  "schema_id": "module2_breakdown_v1"
}
```

**Rule:** choose best **validation** F1; if within 0.01 F1, prefer higher val PR-AUC; if still close (ΔF1≤0.01 and ΔPR-AUC≤0.01), prefer lower `latency_ms_p50`. Never select on test.

## Metrics arena export

```text
metrics/
  module{N}/
    arena_module{N}_11models_v2.json
```

Old `arena_module*_11models.json` → quarantine (see QUARANTINE_MOVE_LIST.md).

## joblib payload shape (ML)

```python
{
  "model": <fitted_estimator>,
  "model_key": "...",
  "schema_id": "...",
  "feature_names": [...],
  "label_name": "...",
  "artifact_version": "1.0"
}
```

## PyTorch payload

- `.pt` = `state_dict` or `{ "state_dict", "model_key", "arch_config" }`
- Companion `.meta.json` mandatory
- Norm stats in `.norm.npz` with keys `mean`, `std` fitted on **train only**
