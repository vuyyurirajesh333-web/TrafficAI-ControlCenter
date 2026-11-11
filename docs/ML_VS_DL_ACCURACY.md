# Why Classical ML Beat Deep Learning in Your Colab Runs

**Short answer:** Yes — on *these* tables it is normal that Random Forest / XGBoost / Gradient Boosting show higher F1 than LSTM/Transformer/TCN. That does **not** mean tree models are “better AI” for traffic in the real world. It usually means the **labels were easy to recover from the same features the rules used**, or that a **leaky feature** told the model the answer.

Your arena numbers (approx.):

| Module | Best classical | Best DL (typical) | Red flag |
|--------|----------------|-------------------|----------|
| Module 1 Bus priority | RF / GB ≈ **0.999** F1 | TCN ≈ 0.94 | Near-perfect on rule-derived labels |
| Module 2 Breakdown | XGBoost ≈ **0.95** F1 | TCN ≈ 0.83 | `has_any_anomaly_in_video` leaks label |
| Module 3 U-turn | RF / GB / XGB = **1.000** F1 | Transformers ≈ 0.65–0.74 | Perfect score ⇒ overfit / leakage / tiny eval |

Perfect or 0.999 F1 on held-out data is almost never “you built a genius model.” It means **the test was too easy or contaminated**.

---

## What each module was actually learning

### Module 1 — Bus priority (`EXTEND_GREEN` / `TRUNCATE_RED` / `NONE`)

- Features include `dist_to_junction_m`, `speed_mps`, **`eta_sec`**, `lateness_sec`, …
- Training meta note: *labels from a **rule engine** on real features*.
- In the data, `EXTEND_GREEN` rows have ETA ~20–40s; `TRUNCATE_RED` ~40–70s; `NONE` has huge ETAs.
- Trees literally re-learn: *if eta is small and distance is small → priority*. That recovers the label with near 100% F1.
- Deep sequence models are a poor fit for flat tabular GPS/GTFS rows and will look “worse” even when they are healthier for harder future tasks.

**You did not necessarily mess up training code.** You evaluated models on a **synthetic decision that is a function of the inputs**.

### Module 2 — Breakdown / anomaly windows

- Features are OpenCV motion stats: `motion_mean`, `stationary_score`, `rule_stop_hint`, …
- Feature list also includes **`has_any_anomaly_in_video`**.
- Correlation of that flag with `y` ≈ **0.75**. When the flag is 0, **all windows are negative**. That is classic **leakage**: the model can learn “this video is an anomaly video” instead of “this window looks like a stoppage.”
- Frame-difference features on whole frames also confuse **busy vs quiet scenes** with **vehicle breakdowns**. Your ops pipeline now uses **track-level sudden stop** (moved → stopped + moving peers), which matches the Module 2 design doc better than whole-frame motion ML.

### Module 3 — U-turn

- Labels: *NGSIM heading-reversal heuristic* (`uturn_candidate`).
- Features: `turn_intensity`, `sum_abs_dheading`, `tortuosity`, … — the same geometry that defined the label.
- F1 = **1.0** for three tree models is a strong sign of **label ≈ feature function** and/or **train/test contamination** (same trajectories / vehicles leaking across folds).

---

## Is it normal that ML > DL here?

**Yes, for tabular, small/medium, rule-ish problems:**

1. **Gradient-boosted trees** dominate classical tabular benchmarks.
2. **DL needs more data, careful regularization, and a representation that matches the task** (raw video / long tracks), not 8–16 hand features.
3. When the label is a **threshold on ETA / heading / motion**, trees win; DL cannot invent a harder problem than you gave it.

**No, it is not proof that DL “failed” for the government product.** The live app uses **YOLO + ByteTrack + decision rules** for perception. The Colab arena is a **feature-table bake-off**, not the same system operators use when they upload CCTV.

---

## What to retrain (recommended order)

### Must-fix before claiming production metrics

1. **Remove leaky features**
   - Module 2: drop `has_any_anomaly_in_video`, and prefer not to train on `rule_stop_hint` if that rule also defines positives.
   - Module 1: if the product decision is rule-based on ETA, **do not train a classifier on ETA to predict the same rule**. Either:
     - keep the rule as the decision engine, or
     - predict something the rule does **not** already compute (e.g. true arrival delay vs schedule, passenger load, downstream queue).
   - Module 3: label U-turns from **independent annotation** (or SUMO / inD), not from the same heading stats used as features.

2. **Proper splits**
   - Split by **video_id / day / vehicle_id / route**, never random rows.
   - Report metrics on a **held-out corridor** never seen in training.

3. **Sane baselines**
   - Always publish the **rule-only** F1 next to ML F1. If ML ≈ rule, you are not adding value yet.

4. **Class imbalance**
   - Module 1 is ~415k NONE vs ~4k actions. Report **per-class precision/recall**, not only micro-F1.

### Nice-to-have for a stronger thesis / demo

5. Retrain Module 2 on **track features**: prior max speed, stop duration, peer moving count, lane ROI — matching the live perception engine.
6. Use DL on **raw clips or track sequences** (ByteTrack trajectories), not on 12 motion scalars.
7. Calibrate probabilities (`predict_proba`) and set an **ops threshold** (e.g. alert only if p ≥ 0.8 and stop ≥ 4s).

### What you can keep

- YOLO pretrained + ByteTrack for detection/tracking — correct for all three modules.
- Tree models as **tabular decision aids** *after* leakage is removed.
- Arena JSON for engineering — **hide from the operator UI** (already done in the ops console).

---

## How to talk about this in a viva / report

> “Classical models scored higher because labels were generated from rules and features that overlap those rules. Near-perfect F1 indicates label leakage / easy tabular recovery, not superiority of ML over DL for CCTV understanding. Production behaviour relies on YOLO+ByteTrack perception with explicit sudden-stop and priority logic; model tables will be retrained without leaky features and with video-level splits.”

That statement is accurate and stronger than claiming “XGBoost beats Transformers.”

---

## Checklist before next Colab run

- [ ] Drop `has_any_anomaly_in_video` from Module 2 feature matrix  
- [ ] GroupKFold / split by `video_id` (M2) and `vehicle_id` or time (M3)  
- [ ] Module 1: either remove `eta_sec` from features **or** change the prediction target  
- [ ] Module 3: independent U-turn labels; do not threshold `turn_intensity` to create `y` then train on `turn_intensity`  
- [ ] Log rule-baseline F1 beside every model  
- [ ] Export confusion matrices + calibration plots  
- [ ] Re-run arena JSON into `metrics/` and keep ops UI free of model comparison tables  
