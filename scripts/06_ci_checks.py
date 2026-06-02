#!/usr/bin/env python
"""Lightweight repository checks for pre-push and GitHub Actions.

These checks intentionally avoid importing torch/torchvision so CI stays fast.
They validate project structure, Python syntax, notebook JSON, and a few
competition-critical guardrails.
"""

from __future__ import annotations

import json
import py_compile
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "AGENT_INSTRUCTIONS.md",
    "requirements.txt",
    "configs/baseline_frcnn.yaml",
    "configs/v2_three_model.yaml",
    "scripts/00_preflight.py",
    "scripts/01_train_scanner.py",
    "scripts/02_predict_scanner.py",
    "scripts/03_make_submission.py",
    "scripts/04_full_pipeline.py",
    "scripts/05_audit_dimensions.py",
    "scripts/07_train_global_classifier.py",
    "scripts/08_train_crop_classifier.py",
    "scripts/09_full_pipeline_v2.py",
    "scripts/10_predict_global_classifier.py",
    "scripts/11_predict_crop_classifier.py",
    "scripts/12_fuse_predictions_v2.py",
    "notebooks/LG_CXR_FRCNN_Colab.ipynb",
    "notebooks/LG_CXR_FRCNN_Colab_Standalone.ipynb",
    "notebooks/LG_CXR_FRCNN_Kaggle.ipynb",
    "notebooks/LG_CXR_FRCNN_Kaggle_V2_Three_Model.ipynb",
    "src/data/image_sizes.py",
]

README_REQUIRED_PHRASES = [
    "Never train class `14` as an object",
    "original train.csv boxes -> PNG image boxes -> Faster R-CNN",
    "Faster R-CNN PNG-space boxes -> original-space boxes -> submission.csv",
    "python scripts/05_audit_dimensions.py --config configs/baseline_frcnn.yaml",
]

BANNED_CODE_PATTERNS = [
    re.compile(r"^\s*(from|import)\s+ultralytics\b", re.MULTILINE),
    re.compile(r"^\s*(from|import)\s+yolov?\d*\b", re.MULTILINE | re.IGNORECASE),
]


def main() -> int:
    failures: list[str] = []
    failures.extend(check_required_files())
    failures.extend(check_python_syntax())
    failures.extend(check_notebooks())
    failures.extend(check_readme_guardrails())
    failures.extend(check_banned_code_imports())
    failures.extend(check_standalone_notebook_embeds_current_files())

    if failures:
        print("\nCI checks failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("CI checks passed.")
    return 0


def check_required_files() -> list[str]:
    failures = []
    for relative_path in REQUIRED_FILES:
        if not (PROJECT_ROOT / relative_path).exists():
            failures.append(f"Missing required file: {relative_path}")
    return failures


def check_python_syntax() -> list[str]:
    failures = []
    for path in sorted(PROJECT_ROOT.rglob("*.py")):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            failures.append(f"Python syntax error in {rel(path)}: {exc.msg}")
    return failures


def check_notebooks() -> list[str]:
    failures = []
    for path in sorted((PROJECT_ROOT / "notebooks").glob("*.ipynb")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"Notebook JSON invalid in {rel(path)}: {exc}")
            continue
        if payload.get("nbformat") != 4:
            failures.append(f"Notebook nbformat must be 4: {rel(path)}")
        if not payload.get("cells"):
            failures.append(f"Notebook has no cells: {rel(path)}")
    return failures


def check_readme_guardrails() -> list[str]:
    failures = []
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    for phrase in README_REQUIRED_PHRASES:
        if phrase not in readme:
            failures.append(f"README missing guardrail phrase: {phrase}")
    return failures


def check_banned_code_imports() -> list[str]:
    failures = []
    for path in sorted(PROJECT_ROOT.rglob("*.py")):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in BANNED_CODE_PATTERNS:
            if pattern.search(text):
                failures.append(f"Banned YOLO/Ultralytics import found in {rel(path)}")
    return failures


def check_standalone_notebook_embeds_current_files() -> list[str]:
    failures = []
    path = PROJECT_ROOT / "notebooks" / "LG_CXR_FRCNN_Colab_Standalone.ipynb"
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"Cannot inspect standalone notebook: {exc}"]

    source = "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))
    for relative_path in [
        "src/data/image_sizes.py",
        "scripts/05_audit_dimensions.py",
        "scripts/06_ci_checks.py",
    ]:
        if relative_path not in source:
            failures.append(f"Standalone notebook does not embed or mention {relative_path}")
    return failures


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


if __name__ == "__main__":
    raise SystemExit(main())
