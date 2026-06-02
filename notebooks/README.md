# Notebooks

`LG_CXR_FRCNN_Colab.ipynb` is the runnable workflow for Colab when the project folder already exists in Drive or has been uploaded to Colab.

`LG_CXR_FRCNN_Colab_Standalone.ipynb` is the copy/paste workflow for Colab when the user does not have the project folder yet. It bootstraps the project files into `/content/amia-lgcxr-frcnn`.

`LG_CXR_FRCNN_Kaggle.ipynb` is the Kaggle Notebook workflow. It clones the GitHub repo into `/kaggle/working/Amia_keagle_challenge`, uses `/kaggle/input/amia-public-challenge-2026`, and writes `submission.csv` to `/kaggle/working`.

`LG_CXR_FRCNN_Kaggle_V2_Three_Model.ipynb` is the complete Kaggle V2 workflow with three models: Faster R-CNN scanner, global ResNet18 classifier, and crop ResNet18 verifier. It uses `configs/v2_three_model.yaml`.

Both notebooks include a dimension-audit step:

```bash
python scripts/05_audit_dimensions.py --config configs/baseline_frcnn.yaml
```

Run it before changing `scanner.image_size` or `scanner.max_size`.

The notebooks rely on the project's coordinate-space fix: `train.csv` boxes are original scan coordinates, `img_size.csv` gives original dimensions, training targets are scaled into PNG space, and predictions are scaled back to original space for submission.

Keep both notebooks synchronized with README commands and script arguments.
