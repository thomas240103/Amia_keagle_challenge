# LG-CXR FRCNN Baseline: Local-Global Chest X-ray Detection with Faster R-CNN

This project builds a reproducible baseline for the AMIA Public Challenge 2026 using a license-friendlier PyTorch stack. V1 uses `torchvision` Faster R-CNN ResNet50-FPN as the local scanner/detector, writes Kaggle-valid `submission.csv` files, and keeps optional ResNet18 global and crop classifiers as future extensions.

The competition evaluates standard PASCAL VOC 2010 mean Average Precision at IoU > 0.4. This repository therefore prioritizes clean box prediction, correct class mapping, and exact submission formatting before model complexity.

## Why This Project Exists

- Provide a YOLO-free and Ultralytics-free baseline.
- Give humans and agents a clean project that can be reproduced from the README.
- Run preflight checks before spending GPU time.
- Produce leaderboard-valid output even when no detections survive filtering.
- Make the first working path Colab-friendly while preserving Kaggle-style paths.

## Model Idea

V1 is:

```text
Faster R-CNN ResNet50-FPN -> filtered detections -> submission.csv
```

Planned optional extensions are:

- ResNet18 global classifier: predicts image-level pathology probabilities and can reweight detector scores later.
- ResNet18 crop classifier: verifies candidate detection crops later.
- Fusion logic: V1 is pass-through; learned fusion is intentionally deferred.

Optional classifier checkpoints may be absent. The code must continue with Faster R-CNN only.

## Theory

Object detection predicts both what is present and where it appears. Classification predicts what is present in the whole image, but not the box location. Chest X-ray pathology tasks often need detection because the leaderboard expects class IDs, confidence scores, and bounding boxes.

Faster R-CNN is a two-stage detector. The first stage, the Region Proposal Network, proposes likely object regions. The second stage classifies those proposals and refines their bounding boxes. The FPN backbone helps the detector reason across scales, which is useful for medical findings that may be small, diffuse, or low contrast.

Faster R-CNN is a good first medical detection baseline because it is widely available in `torchvision`, avoids YOLO/Ultralytics licensing concerns, is simple to checkpoint, and can be trained with ordinary PyTorch loops.

Global image-level classification can help later because some findings may be visually subtle but globally likely. Crop-level verification can help later by rescoring candidate boxes after the detector proposes them. These are not required for V1.

Class `14` is `"No finding"`, but it is not a physical object. Training the detector to localize class `14` would teach it a fake box target. Instead, class `14` rows are treated as negative/no-finding annotations, and class `14` appears only as the submission fallback.

Validation should use IoU threshold `0.4` because the challenge uses VOC mAP at IoU > 0.4.

## Competition Rules

Submission columns must be exactly:

```text
image_id,PredictionString
```

Each `PredictionString` contains repeated groups:

```text
class_id confidence xmin ymin xmax ymax
```

Real pathology classes are original competition IDs `0..13`. Class ID `14` is `"No finding"` fallback only.

If a test image has no detections, output exactly:

```text
14 1.0 0 0 1 1
```

Never output an empty `PredictionString`.

Torchvision detector labels use `0` for background, so the mapping is:

```text
competition 0 -> detector 1
competition 1 -> detector 2
...
competition 13 -> detector 14
```

Detector predictions are mapped back to competition IDs `0..13` before submission.

## Environment

Kaggle defaults:

```text
DATA_ROOT=/kaggle/input/amia-public-challenge-2026
WORK_DIR=/kaggle/working
```

Expected competition dataset layout:

```text
DATA_ROOT/
  train/
  test/
  img_size.csv
  sample_submission.csv
  test.csv
  train.csv
```

The code now prefers these exact files and folders before falling back to heuristic discovery. Training image IDs come from the `train/` folder when it exists, so train images without valid object rows can still be used as negative/no-finding Faster R-CNN samples. Test prediction prefers IDs from `test.csv`, while final submission still uses `sample_submission.csv` as the source of truth for row order and required IDs.

`img_size.csv` is used by preflight to infer image dimensions and report whether object boxes fit within the declared image width and height. The image resolver accepts common ID variants such as bare stems, filenames, and path-like strings.

Important coordinate-space rule:

- `train.csv` bounding boxes are in original scan coordinate space.
- `img_size.csv` stores original dimensions per image. In this dataset, `dim0` is original height and `dim1` is original width.
- The PNG files in `train/` and `test/` may be resized, commonly around `1024x1024`.
- Faster R-CNN training targets must be in the actual PNG tensor coordinate space.
- Kaggle submission boxes must be written back in original coordinate space.

The code therefore performs two conversions:

```text
training:   original train.csv boxes -> PNG image boxes -> Faster R-CNN
inference:  Faster R-CNN PNG-space boxes -> original-space boxes -> submission.csv
```

Do not divide coordinates by a constant like `1024`. Original dimensions vary per image.

Colab defaults used in the notebook:

```text
PROJECT_DIR=/content/drive/MyDrive/amia-lgcxr-frcnn
DATA_ROOT=/content/drive/MyDrive/datasets/amia-public-challenge-2026
WORK_DIR=/content/drive/MyDrive/amia-lgcxr-frcnn/outputs
```

GPU is recommended. Torch and torchvision are required. `timm` is optional for future classifier work. YOLO, Ultralytics, and RT-DETR are not used for V1.

Runtime path overrides are supported through CLI flags or environment variables:

```bash
python scripts/00_preflight.py --config configs/baseline_frcnn.yaml --data-root "$DATA_ROOT" --work-dir "$WORK_DIR"
```

or:

```bash
export LGCXR_DATA_ROOT=/path/to/dataset
export LGCXR_WORK_DIR=/path/to/outputs
```

## Colab Setup

Open:

```text
notebooks/LG_CXR_FRCNN_Colab.ipynb
```

Use this notebook when the full `amia-lgcxr-frcnn/` project folder already exists in Google Drive or has been uploaded to Colab. It includes cells to mount Google Drive, locate the project, install requirements, check CUDA, configure dataset and output paths, run preflight, run smoke test, train, predict, and make a submission.

If you plan to copy and paste a notebook into Colab without uploading the project folder, use the standalone bootstrap notebook instead:

```text
notebooks/LG_CXR_FRCNN_Colab_Standalone.ipynb
```

The standalone notebook creates the project files under:

```text
/content/amia-lgcxr-frcnn
```

Then it installs dependencies and runs the same preflight, smoke test, training, inference, and submission commands. This is the safer notebook when you do not have the folder in Drive yet.

Place the dataset at:

```text
/content/drive/MyDrive/datasets/amia-public-challenge-2026
```

or adjust `DATA_ROOT` in the notebook. All important Colab outputs are saved to `WORK_DIR`, including checkpoints, predictions, debug images, training summaries, and `submission.csv`.

## Step-by-Step Usage

Run preflight first:

```bash
python scripts/00_preflight.py --config configs/baseline_frcnn.yaml
```

Audit image dimensions and box scale:

```bash
python scripts/05_audit_dimensions.py --config configs/baseline_frcnn.yaml
```

Train the scanner:

```bash
python scripts/01_train_scanner.py --config configs/baseline_frcnn.yaml
```

Predict validation and test:

```bash
python scripts/02_predict_scanner.py --config configs/baseline_frcnn.yaml
```

Create the final submission:

```bash
python scripts/03_make_submission.py --config configs/baseline_frcnn.yaml
```

Single-command pipeline:

```bash
python scripts/04_full_pipeline.py --config configs/baseline_frcnn.yaml
```

Smoke test:

```bash
python scripts/04_full_pipeline.py --config configs/baseline_frcnn.yaml --smoke-test
```

Smoke mode uses small subsets, trains for at most one epoch, and creates a format-valid submission. It is for plumbing checks, not score.

## Preflight

Preflight must run before training. It checks:

- Exact Kaggle layout files and folders when present.
- Dataset root exists.
- CSV files load.
- Train CSV is identified.
- Test CSV is identified.
- `img_size.csv` is identified when present.
- Sample submission is identified.
- Train and test folders contain images.
- Train CSV image IDs resolve inside `train/`.
- Test CSV and sample submission image IDs resolve inside `test/`.
- Required columns are inferred.
- 50 train images open when available.
- 20 test images open when available.
- Bounding boxes are valid for object classes.
- Bounding boxes are checked against `img_size.csv` when possible.
- Class mapping is valid.
- Class `14` is handled as no-finding, not as a detection object.
- Sample submission format is understood.
- A small debug image is saved.
- Training does not start if critical checks fail.

Preflight writes:

```text
WORK_DIR/lgcxr_preflight_status.json
WORK_DIR/debug_preflight_annotation.png
```

## Dimension Audit

Before changing `scanner.image_size`, run:

```bash
python scripts/05_audit_dimensions.py --config configs/baseline_frcnn.yaml
```

This writes:

```text
WORK_DIR/lgcxr_dimension_audit.json
```

The audit studies:

- Dimensions declared in `img_size.csv`.
- Real PIL image dimensions from `train/` and `test/`.
- Whether original image dimensions differ from PNG dimensions.
- Whether `train.csv` IDs resolve to files in `train/`.
- Whether `test.csv` IDs resolve to files in `test/`.
- Box width, height, and area in original pixels.
- Box width, height, and area as fractions of image size.
- Box size after scaling from original coordinates to actual PNG coordinates.
- Estimated box dimensions after the current torchvision resize.
- Whether boxes look normalized instead of pixel-based.
- Whether boxes fall outside declared image dimensions.

Current V1 config:

```yaml
scanner:
  image_size: 800
  max_size: 1200
```

In torchvision Faster R-CNN, this means the shortest side is resized to about `800`, unless that would make the longest side exceed `1200`. Aspect ratio is preserved. This is a conservative first baseline for GPU memory.

How to interpret the report:

- If most images are around `1024x1024`, `800/1200` is usually reasonable for V1.
- If most images are `2048x2048` or larger and many boxes become under about `8-12` pixels wide after resizing, try `1024/1536` or `1200/1800`.
- If boxes look normalized, fix coordinate parsing before training.
- If many boxes exceed image dimensions, inspect whether the CSV uses width/height boxes, pixel boxes, normalized boxes, or a different coordinate origin.
- If `boxes_scaled_original_to_png` is nonzero, that is expected for this dataset and means the original-space fix is active.
- If CUDA OOM occurs, keep `800/1200` or reduce batch size before raising dimensions.

## Training

The primary detector is `torchvision` Faster R-CNN ResNet50-FPN. The builder prefers `fasterrcnn_resnet50_fpn_v2` when available and falls back to `fasterrcnn_resnet50_fpn`.

When `DATA_ROOT/train/` exists, training and validation splits are made from the images in that folder, not only from the IDs that appear in `train.csv`. Images with no valid class `0..13` boxes are valid negative samples. Annotation rows with class `14` are ignored as objects.

Because `train.csv` boxes are original-space boxes, `CXRDetectionDataset` uses `img_size.csv` to scale every object box from original scan dimensions into the actual PNG size before passing targets to Faster R-CNN. This prevents clipping/degenerate labels when original coordinates are larger than the PNG dimensions.

Checkpoint files:

```text
WORK_DIR/lgcxr_scanner_fasterrcnn_best.pth
WORK_DIR/lgcxr_scanner_fasterrcnn_last.pth
```

If `scanner.resume: true`, training resumes from the last checkpoint when it exists. AMP is used only when CUDA is available. The validation proxy uses VOC-style matching at IoU `0.4`. Outputs are saved under `WORK_DIR`.

Batch size is intentionally small by default because Faster R-CNN is memory-heavy. If CUDA OOM occurs, reduce `scanner.batch_size` or `scanner.image_size`.

Do not change `scanner.image_size` blindly. Run the dimension audit first, then choose the next experiment based on resized box-size statistics.

## Inference

Inference loads the best scanner checkpoint, predicts validation and test images, applies confidence thresholding, applies class-wise NMS, maps internal labels back to original class IDs, and saves:

```text
WORK_DIR/lgcxr_fast_val_predictions.csv
WORK_DIR/lgcxr_fast_test_predictions.csv
```

Prediction rows contain:

```text
image_id,class_id,confidence,xmin,ymin,xmax,ymax
```

For test inference, `test.csv` image IDs are preferred when available. Submission creation still matches predictions back to `sample_submission.csv` IDs using tolerant ID variants, so `abc123`, `abc123.png`, and path-like image IDs can still line up.

When `img_size.csv` is available, scanner predictions are saved in original coordinate space after scaling from PNG/model image coordinates back to the original scan dimensions. This is the coordinate space expected by `submission.csv`.

## Submission

Submission creation uses the sample submission as the source of truth for row order and test IDs. It asserts:

- Columns are exactly `["image_id", "PredictionString"]`.
- Row count equals sample submission.
- No missing values exist.
- Every prediction group length is a multiple of 6.
- Class IDs are in `0..14`.
- Class `14` appears only alone as the exact no-finding fallback.

Final output:

```text
WORK_DIR/submission.csv
```

On Kaggle this resolves to:

```text
/kaggle/working/submission.csv
```

## Reproducibility

Reproducibility controls:

- Central config: `configs/baseline_frcnn.yaml`
- Seed: `42`
- Stable train/validation split from the config seed.
- Checkpoint paths recorded in config.
- Preflight status saved before training.
- Training summary JSON saved after training.

Rerun from a clean environment by installing requirements, setting paths, running preflight, running smoke test, then running the full pipeline.

## Troubleshooting

Dataset not found: check `data_root`, `LGCXR_DATA_ROOT`, or the notebook `DATA_ROOT`.

Sample submission not found: verify the dataset contains a CSV with `image_id` and `PredictionString`.

Required columns not inferred: inspect the train CSV and update `src/data/columns.py` with the dataset's column names.

Invalid boxes: check annotation rows with negative coordinates, swapped corners, missing class IDs, or zero-area boxes.

Too many unreadable images: verify image file extensions and dataset placement.

CUDA not available: training still runs on CPU, but it will be slow.

OOM: reduce `scanner.batch_size`, reduce `scanner.image_size`, or enable smoke mode.

Checkpoint missing: run training before inference, or verify `scanner.best_ckpt` points to an existing file.

Submission format mismatch: run `scripts/03_make_submission.py`; it includes strict validation assertions.

## Roadmap

- Train longer.
- Tune thresholds per class.
- Add global ResNet18 classifier.
- Add crop ResNet18 classifier.
- Add learned fusion.
- Add hard negative mining.
- Add pseudo-labeling if competition rules allow.
- Add WBF/ensembling.
- Add more precise VOC mAP evaluation.

## Agent / Codex / Claude Workflow Rules

Every agent working on this repository must:

1. Read `README.md` before making changes.
2. Read `AGENT_INSTRUCTIONS.md` before making changes.
3. Check the current project structure before editing.
4. Keep the README consistent with the code.
5. Update README whenever behavior, commands, config, outputs, checkpoints, model choice, or submission logic changes.
6. Never silently change the submission format.
7. Never train class `14` as an object.
8. Never output empty `PredictionString`.
9. Always run preflight before training.
10. Prefer small, testable changes.
11. Record important changes in a README changelog section.
12. Do not introduce YOLO/Ultralytics into this project.
13. Do not over-engineer before a valid submission exists.
14. Make every script runnable from the command line.
15. Keep config centralized in YAML.
16. Keep `notebooks/LG_CXR_FRCNN_Colab.ipynb` synchronized with script commands.
17. Keep `notebooks/LG_CXR_FRCNN_Colab_Standalone.ipynb` synchronized when project files or script commands change.
18. If a script command changes, update README, `AGENT_INSTRUCTIONS.md`, and both Colab notebooks.

## Changelog

- 2026-06-02: Initial V1 project scaffold with Colab-first workflow, Faster R-CNN baseline, preflight, training, inference, and submission scripts.
- 2026-06-02: Added standalone Colab bootstrap notebook guidance for copy/paste use without uploading the project folder first.
- 2026-06-02: Hardened dataset layout handling for explicit `train/`, `test/`, `train.csv`, `test.csv`, `img_size.csv`, and `sample_submission.csv`; training now includes folder-level negative images and preflight cross-checks CSV/image alignment.
- 2026-06-02: Added `scripts/05_audit_dimensions.py` and explicit `scanner.max_size` so image resize decisions can be based on real image and box statistics.
- 2026-06-02: Fixed original-coordinate bounding-box handling: train boxes are scaled from original scan space into PNG space for Faster R-CNN, and inference boxes are scaled back to original space for submission.
