"""Image file indexing and resolution."""

from __future__ import annotations

from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_image_files(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def image_id_from_path(path: str | Path) -> str:
    return Path(path).stem


def image_id_variants(image_id: str) -> set[str]:
    raw = str(image_id).strip().strip('"').strip("'")
    normalized = raw.replace("\\", "/")
    path = Path(normalized)
    candidates = {
        raw,
        normalized,
        path.name,
        path.stem,
    }
    return {candidate.lower() for candidate in candidates if candidate}


def image_ids_from_paths(paths: list[Path]) -> list[str]:
    seen: set[str] = set()
    image_ids: list[str] = []
    for path in sorted(paths):
        image_id = image_id_from_path(path)
        key = image_id.lower()
        if key not in seen:
            seen.add(key)
            image_ids.append(image_id)
    return image_ids


def build_image_index(paths: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in paths:
        keys = image_id_variants(str(path)) | image_id_variants(path.name) | image_id_variants(path.stem)
        for key in keys:
            index.setdefault(key, path)
    return index


def resolve_image_path(image_id: str, image_index: dict[str, Path]) -> Path | None:
    for candidate in image_id_variants(image_id):
        if candidate in image_index:
            return image_index[candidate]
    return None
