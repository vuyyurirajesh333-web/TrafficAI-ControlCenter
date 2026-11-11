# Student Google Drive Checklist (Today) — Phase A

**Goal:** Verify folders and inventory. **Do not download multi-GB AI City yet** unless files are already missing and you only need to *confirm* presence.  
**Do not retrain today.**

## Root folder (exact name)

```text
Google Drive / MyDrive / TrafficAI_ControlCenter/
```

If missing, create that exact folder name.

## Folders to create/verify (exact names)

```text
TrafficAI_ControlCenter/
├── datasets/
│   ├── raw/
│   │   ├── delhi_gtfs/
│   │   ├── delhi_bus_gps/
│   │   ├── aicity2021_t4/
│   │   ├── ngsim/
│   │   ├── india_detection/
│   │   └── osm_india/
│   ├── processed/
│   ├── splits/
│   │   ├── module1/
│   │   ├── module2/
│   │   └── module3/
│   └── quarantine_leaky/
├── notebooks/
├── models/
│   ├── module1/
│   ├── module2/
│   ├── module3/
│   └── quarantine_leaky/
├── metrics/
│   └── quarantine_leaky/
├── evaluation/
│   ├── module1/
│   ├── module2/
│   └── module3/
└── demos/
    └── sample_videos/
```

## Verify checklist (tick on paper)

1. [ ] `TrafficAI_ControlCenter` exists on Drive  
2. [ ] `datasets/raw/delhi_gtfs/` — has GTFS txt/zip or extracted files  
3. [ ] `datasets/raw/delhi_bus_gps/` — has GPS archives/csv/sqlite  
4. [ ] `datasets/raw/aicity2021_t4/` — note whether `.mp4` videos exist (count approx.)  
5. [ ] Write down: AI City video count = ______ ; folder size ≈ ______ GB  
6. [ ] `datasets/raw/ngsim/` — present or empty?  
7. [ ] `datasets/raw/india_detection/` — present or empty? (P2 only)  
8. [ ] `datasets/quarantine_leaky/` created  
9. [ ] `models/quarantine_leaky/` created  
10. [ ] `metrics/quarantine_leaky/` created  
11. [ ] Confirm Colab can mount Drive and see `MyDrive/TrafficAI_ControlCenter`  
12. [ ] **Do not** start AI City re-download today if videos already exist  
13. [ ] **Do not** delete old models/metrics — only prepare quarantine folders  

## If AI City videos are MISSING

Stop after noting “MISSING”. Next phase (Phase B) will give download steps from:  
https://www.aicitychallenge.org/2021-track4-download/  

Do **not** start the huge download until Phase B unless you have free time/space and want to begin overnight — Phase A only requires **verification**.

## Sync note

Local repo already has processed GTFS/GPS and anomaly_events CSV. Drive should hold **raw** videos/archives that local checkout lacks.
