# Agent Instructions

This repository is a clean V1 Faster R-CNN baseline for the AMIA Public Challenge 2026. Keep it reproducible, leaderboard-valid, and Colab-friendly.

Every agent working on this repository must:

1. Read `README.md` before making changes.
2. Read this file before making changes.
3. Check the current project structure before editing.
4. Keep the README consistent with the code.
5. Keep `notebooks/LG_CXR_FRCNN_Colab.ipynb` synchronized with script commands.
6. Keep `notebooks/LG_CXR_FRCNN_Colab_Standalone.ipynb` synchronized with project files and script commands.
7. Update README, this file, and both notebooks when any script command changes.
8. Update README whenever behavior, commands, config, outputs, checkpoints, model choice, or submission logic changes.
9. Never silently change the submission format.
10. Never train class `14` as an object.
11. Never output an empty `PredictionString`.
12. Always run preflight before training.
13. Prefer small, testable changes.
14. Record important changes in the README changelog.
15. Do not introduce YOLO or Ultralytics.
16. Do not introduce RT-DETR for the first baseline.
17. Do not over-engineer before a valid submission exists.
18. Make every script runnable from the command line and from Colab.
19. Keep config centralized in YAML.
20. Keep the V2 Kaggle notebook synchronized with V2 scripts/config.
21. Run `python scripts/06_ci_checks.py` before pushing.

Operational notes:

- Main detector: torchvision Faster R-CNN ResNet50-FPN.
- Prefer explicit competition files and folders: `train/`, `test/`, `train.csv`, `test.csv`, `img_size.csv`, and `sample_submission.csv`.
- Training IDs should come from the `train/` folder when available so folder-level negative images are not lost.
- Test prediction may use `test.csv`, but final submission row order must come from `sample_submission.csv`.
- Run `scripts/05_audit_dimensions.py` before changing `scanner.image_size` or `scanner.max_size`.
- Treat `train.csv` boxes as original scan coordinates. Use `img_size.csv` to scale targets into PNG space for training and predictions back to original space for submission.
- Keep `.github/workflows/ci.yml`, `.githooks/pre-push`, and `scripts/06_ci_checks.py` aligned with project guardrails.
- Detector labels: background is `0`; competition classes `0..13` map to internal labels `1..14`.
- Class `14` is the no-finding fallback only and is ignored during detector training.
- Empty test detections must become exactly `14 1.0 0 0 1 1`.
- Primary output: `/kaggle/working/submission.csv` on Kaggle, or `WORK_DIR/submission.csv` in Colab.
- Optional global and crop classifiers must never block V1 submission.
