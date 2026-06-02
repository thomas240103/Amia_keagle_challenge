# Notebooks

`LG_CXR_FRCNN_Colab.ipynb` is the runnable workflow for Colab when the project folder already exists in Drive or has been uploaded to Colab.

`LG_CXR_FRCNN_Colab_Standalone.ipynb` is the copy/paste workflow for Colab when the user does not have the project folder yet. It bootstraps the project files into `/content/amia-lgcxr-frcnn`.

Both notebooks include a dimension-audit step:

```bash
python scripts/05_audit_dimensions.py --config configs/baseline_frcnn.yaml
```

Run it before changing `scanner.image_size` or `scanner.max_size`.

The notebooks rely on the project's coordinate-space fix: `train.csv` boxes are original scan coordinates, `img_size.csv` gives original dimensions, training targets are scaled into PNG space, and predictions are scaled back to original space for submission.

Keep both notebooks synchronized with README commands and script arguments.
