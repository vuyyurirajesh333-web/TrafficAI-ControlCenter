# PHASE A — Architecture Freeze

**Date:** 2026-11-10  
**Status:** CONTRACTS FROZEN — awaiting student `PHASE A DONE`

## Approved decisions (locked)

1. Arena roster: LR, RF, XGBoost, LightGBM + MLP, LSTM, GRU, BiLSTM, TCN, Transformer Encoder, Temporal CNN. Removed GB / GAT_GRU / Decision_Policy_Head.  
2. Module 1 labels: hybrid human CCTV + GPS/GTFS weak labels; never ETA/distance threshold labels.  
3. Rules fallback default OFF; warning required if enabled.  
4. AI City T4 raw videos: verify/restore on Drive for Module 2.  
5. Keep YOLOv8n pretrained; India fine-tune = P2.  
6. Winner = best val F1 → PR-AUC → lower latency if close.  
7. Agent order after freeze: A → B → C → D → E → F.

## Frozen artifacts (read these)

| Artifact | Path |
|---|---|
| Module 1 feature schema | `config/schemas/module1_features.schema.json` |
| Module 2 feature schema | `config/schemas/module2_features.schema.json` |
| Module 3 feature schema | `config/schemas/module3_features.schema.json` |
| Artifact naming | `docs/architecture_freeze/ARTIFACT_NAMING.md` |
| API contract | `docs/architecture_freeze/API_CONTRACT.md` |
| Quarantine move list | `docs/architecture_freeze/QUARANTINE_MOVE_LIST.md` |
| Drive checklist | `docs/architecture_freeze/STUDENT_DRIVE_CHECKLIST.md` |

## Explicit non-actions in Phase A

- No retraining  
- No multi-GB dataset download required  
- No FastAPI/React rewrite  
- No deletion of old artifacts (quarantine prep only)

## Ownership after you say PHASE A DONE

| Next | Owner |
|---|---|
| Phase B | Agent A — data org + Drive raw verification follow-through |
| Then | A→B→C→D→E→F per master plan |

## Student exit criteria for PHASE A DONE

- [ ] Read the three schema JSON files  
- [ ] Read API + naming docs  
- [ ] Complete Drive checklist (verify only)  
- [ ] Optionally move **dataset** leaky tables locally (metrics/models copy-first)  
- [ ] Reply exactly: `PHASE A DONE`
