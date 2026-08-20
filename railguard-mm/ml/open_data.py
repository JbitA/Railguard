from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

try:
    from download_rail_vivid import REPO as RAIL_VIVID_REPO
    from download_rail_vivid import build_download_manifest, file_sha256
except ModuleNotFoundError:
    from ml.download_rail_vivid import REPO as RAIL_VIVID_REPO
    from ml.download_rail_vivid import build_download_manifest, file_sha256

CATALOG = Path(__file__).resolve().parents[1] / "data" / "open_sources.yaml"
SURFACE_CLASSES = ("Cracks", "Flakings", "Grooves", "Joints", "Shellings", "Spallings", "Squats")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_catalog(path: Path = CATALOG) -> dict:
    data = yaml.safe_load(path.read_text())
    if data.get("version") != 1 or not isinstance(data.get("sources"), dict):
        raise ValueError("unsupported open-data catalog")
    return data


def directory_manifest(root: Path, *, source_id: str, source_meta: dict) -> dict:
    files = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        files.append({"path": rel, "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    if not files:
        raise ValueError(f"no files found under {root}")
    digest = hashlib.sha256(
        "\n".join(f"{x['path']}\t{x['bytes']}\t{x['sha256']}" for x in files).encode("utf-8")
    ).hexdigest()
    return {
        "manifest_version": 1,
        "source_id": source_id,
        "title": source_meta["title"],
        "doi": source_meta["doi"],
        "license": source_meta["license"],
        "file_count": len(files),
        "selection_sha256": digest,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "files": files,
    }


def verify_surface_faults_layout(source: Path) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {}
    lower_dirs = {p.name.lower(): p for p in source.iterdir() if p.is_dir()}
    for class_name in SURFACE_CLASSES:
        class_dir = lower_dirs.get(class_name.lower())
        if class_dir is None:
            raise ValueError(f"missing expected class directory {class_name!r} under {source}")
        images = sorted(p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise ValueError(f"class directory {class_dir} contains no supported images")
        found[class_name] = images
    return found


def import_surface_faults(source: Path, dest: Path, *, copy: bool) -> Path:
    source = source.resolve()
    classes = verify_surface_faults_layout(source)
    dest.mkdir(parents=True, exist_ok=True)
    inventory = []
    for class_name, images in classes.items():
        out_dir = dest / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for path in images:
            out = out_dir / path.name
            if copy:
                shutil.copy2(path, out)
                hash_path = out
            else:
                # Preserve the provider tree when requested; write only an index/manifest.
                hash_path = path
            inventory.append({
                "class": class_name,
                "source_path": str(path),
                "relative_name": f"{class_name}/{path.name}",
                "bytes": hash_path.stat().st_size,
                "sha256": file_sha256(hash_path),
            })
    meta = load_catalog()["sources"]["railway_surface_faults"]
    combined = hashlib.sha256(
        "\n".join(f"{x['class']}\t{x['relative_name']}\t{x['bytes']}\t{x['sha256']}" for x in inventory).encode("utf-8")
    ).hexdigest()
    manifest = {
        "manifest_version": 1,
        "source_id": "railway_surface_faults",
        "title": meta["title"],
        "doi": meta["doi"],
        "license": meta["license"],
        "source_root": str(source),
        "copied_into_repository_data_root": bool(copy),
        "classes": {name: len(paths) for name, paths in classes.items()},
        "image_count": len(inventory),
        "selection_sha256": combined,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "files": inventory,
    }
    path = dest / "surface_faults_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def print_catalog() -> None:
    sources = load_catalog()["sources"]
    print("source_id\tlicense\trole\tdoi")
    for key, value in sources.items():
        print(f"{key}\t{value['license']}\t{value['role']}\t{value['doi']}")


def acquire_rail_vivid(pattern: str, dest: Path, revision: str) -> None:
    # Import lazily so catalog/list/import workflows do not need network access.
    from huggingface_hub import HfApi, hf_hub_download, list_repo_files

    info = HfApi().dataset_info(RAIL_VIVID_REPO, revision=revision)
    resolved = str(info.sha)
    paths = list_repo_files(RAIL_VIVID_REPO, repo_type="dataset", revision=resolved)
    import fnmatch
    selected = [x for x in paths if fnmatch.fnmatch(x, pattern)]
    if not selected:
        raise SystemExit(f"no Rail-VIVID files matched {pattern!r} at revision {resolved}")
    dest.mkdir(parents=True, exist_ok=True)
    downloaded: list[tuple[str, Path]] = []
    for i, repo_path in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {repo_path}")
        local = Path(hf_hub_download(
            RAIL_VIVID_REPO,
            repo_path,
            repo_type="dataset",
            revision=resolved,
            local_dir=dest,
        ))
        downloaded.append((repo_path, local))
    manifest = build_download_manifest(
        requested_revision=revision,
        resolved_revision=resolved,
        pattern=pattern,
        files=downloaded,
    )
    path = dest / "rail_vivid_download_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {path} selection_sha256={manifest['selection_sha256']}")


def main() -> None:
    p = argparse.ArgumentParser(description="Open-data acquisition and provenance entry point")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="show only datasets approved for project experiments")

    acquire = sub.add_parser("acquire", help="download a source with a supported direct acquisition path")
    acquire.add_argument("source", choices=["rail_vivid"])
    acquire.add_argument("--pattern", required=True)
    acquire.add_argument("--revision", default="main")
    acquire.add_argument("--dest", type=Path, default=Path("data/rail_vivid"))

    imp = sub.add_parser("import-surface-faults", help="verify/fingerprint the CC-BY railway surface-fault dataset after provider download")
    imp.add_argument("source_dir", type=Path)
    imp.add_argument("--dest", type=Path, default=Path("data/rail_surface_faults"))
    imp.add_argument("--copy", action="store_true", help="copy images into --dest; default writes a manifest/index only")

    args = p.parse_args()
    if args.command == "list":
        print_catalog()
    elif args.command == "acquire":
        acquire_rail_vivid(args.pattern, args.dest, args.revision)
    else:
        manifest = import_surface_faults(args.source_dir, args.dest, copy=args.copy)
        print(f"verified Railway Track Surface Faults dataset; wrote {manifest}")


if __name__ == "__main__":
    main()
