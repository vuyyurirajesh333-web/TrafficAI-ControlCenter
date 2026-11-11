# Sample CCTV clips for ops testing

| File | Module focus | Notes |
|------|--------------|-------|
| `test_module1_bus.mp4` | Module 1 | Bus in corridor — expect bus priority request |
| `test_module2_breakdown.mp4` | Module 2 | Dense traffic; sudden-stop logic + peer motion |
| `test_module2_sudden_stop_synthetic.mp4` | Module 2 demo | Drawn animation (YOLO may not detect shapes) |
| `test_module3_pexels.mp4` | Module 3 | Real intersection traffic |
| `test_module3_uturn.mp4` | Module 3 / detection | Vehicle trajectories for turn geometry |
| `test_module3_uturn_synthetic.mp4` | Module 3 demo | Drawn U-turn path (for UI rehearsal) |

**Removed duplicates (identical MD5):** `test_module3_intersection.mp4` (= pexels), `test_module3_cars.mp4` (= uturn).

In the ops console, click a sample chip to run `/api/video/analyze-sample` (no manual file pick required). Annotated outputs land in `data/outputs/`.
