# API Contract Freeze — Live Video Analysis (Phase A)

**Status:** FROZEN for Agents C/D  
**Existing routes keep working until Agent C implements this contract.**  
**Perception remains YOLO + ByteTrack. Rules fallback default OFF.**

## Endpoint

`POST /api/video/analyze`

Also apply the same fields to `POST /api/video/analyze-sample` (sample uses JSON body + shared query/form fields as implemented by Agent C).

### Request

**Content-Type:** `multipart/form-data` (upload)  

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `file` | UploadFile | yes (upload path) | — | video |
| `module_mode` | string | no | `all` | `auto` \| `module1` \| `module2` \| `module3` \| `all` |
| `model_module1` | string | no | deployment winner | `model_key` from registry |
| `model_module2` | string | no | deployment winner | `model_key` |
| `model_module3` | string | no | deployment winner | `model_key` |
| `allow_rules_fallback` | bool | no | **`false`** | Must be explicit; never silent |
| `junction_id` | string | no | `junction_01` | existing |
| `max_frames` | int | no | existing default | existing |
| `stride` | int | no | existing default | existing |

**Validation:**
- If `allow_rules_fallback=false` and selected model missing/incompatible → **HTTP 409 or 422** with clear error (no silent rules).
- If `allow_rules_fallback=true` and model fails → allow rules but response must set `decision_source=rules_fallback` and `fallback_warning=true`.

### Response (final shape)

```json
{
  "analysis_id": "optional-sqlite-id",
  "annotated_video_url": "/outputs/annot_....mp4",
  "active_modules": ["module1", "module2", "module3"],
  "alerts": [
    {
      "module": "module2",
      "severity": "high",
      "message": "...",
      "track_id": 18
    }
  ],
  "results": {
    "module1": {
      "selected_model": "lightgbm",
      "model_version": "1.0",
      "schema_id": "module1_bus_priority_v1",
      "model_score": 0.0,
      "decision": "NONE|REQUEST_PRIORITY|STRONG_PRIORITY",
      "decision_source": "model_inference|rules_fallback|error",
      "features_used": {},
      "safety_status": "OK|DEFERRED|REVIEW_REQUIRED|BLOCKED",
      "fallback_warning": false,
      "track_id": null
    },
    "module2": {
      "selected_model": "lstm",
      "model_version": "1.0",
      "schema_id": "module2_breakdown_v1",
      "model_score": 0.0,
      "decision": "NORMAL_OR_EXPECTED_STOP|SUSPECTED_ABNORMAL_STOP",
      "decision_source": "model_inference|rules_fallback|error",
      "features_used": {},
      "safety_status": "OK|REVIEW_REQUIRED|BLOCKED",
      "fallback_warning": false,
      "track_id": 18
    },
    "module3": {
      "selected_model": "tcn",
      "model_version": "1.0",
      "schema_id": "module3_uturn_v1",
      "model_score": 0.0,
      "decision": "NOT_UTURN|UTURN|UTURN_DEMAND_SIGNAL",
      "decision_source": "model_inference|rules_fallback|error",
      "features_used": {},
      "safety_status": "OK|CAUTION|REVIEW_REQUIRED|BLOCKED",
      "fallback_warning": false,
      "track_id": null
    }
  },
  "safety_shield": {
    "overrides": [],
    "blocked_actions": [],
    "notes": []
  },
  "processing_metadata": {
    "perception_engine": "yolov8n_bytetrack",
    "decision_engine": "model_inference_v1",
    "allow_rules_fallback": false,
    "junction_id": "junction_01",
    "frames_processed": 0,
    "schema_versions": {
      "module1": "1.0.0",
      "module2": "1.0.0",
      "module3": "1.0.0"
    }
  }
}
```

Inactive modules may be omitted from `results` or set to `null` when `module_mode` selects a subset.

## Supporting endpoints (Agent C)

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/models/{module}` | List deployable `model_key`s + winner |
| `GET` | `/api/models/{module}/{model_key}` | Artifact meta |
| `GET` | `/api/health` | Must expose `rules_fallback_default: false`, model-loaded flags per module |

## UI contract (Agent D)

- Model dropdowns bind to `/api/models/{module}`.
- If `fallback_warning=true` → **mandatory visible warning banner**.
- Never display fabricated scores; bind only to response fields.
- Show `decision_source` on each module card.

## Non-goals for this contract

- No direct traffic-light actuation.
- No silent substitution of another model if the selected one fails.
