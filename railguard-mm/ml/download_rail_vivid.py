from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, list_repo_files

REPO = "saluslab/Rail-VIVID"


def file_sha256(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_download_manifest(*, requested_revision: str, resolved_revision: str, pattern: str, files: list[tuple[str, Path]]) -> dict:
    entries = [
        {
            "repo_path": repo_path,
            "bytes": local.stat().st_size,
            "sha256": file_sha256(local),
        }
        for repo_path, local in sorted(files, key=lambda item: item[0])
    ]
    combined = hashlib.sha256(
        "\n".join(f"{x['repo_path']}\t{x['bytes']}\t{x['sha256']}" for x in entries).encode("utf-8")
    ).hexdigest()
    return {
        "manifest_version": 1,
        "repository": REPO,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "pattern": pattern,
        "file_count": len(entries),
        "selection_sha256": combined,
        "files": entries,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def main():
    p = argparse.ArgumentParser(description="Pattern-based, revision-recorded Rail-VIVID downloader")
    p.add_argument("--list", action="store_true", help="list repository paths")
    p.add_argument("--pattern", default=None, help="fnmatch path pattern, e.g. 'AtoB_*/*csv'")
    p.add_argument("--revision", default="main", help="HF revision/commit. The exact resolved commit is recorded in the manifest.")
    p.add_argument("--dest", type=Path, default=Path("data/rail_vivid"))
    p.add_argument("--manifest", type=Path, default=None, help="output provenance manifest; defaults inside --dest")
    args = p.parse_args()

    info = HfApi().dataset_info(REPO, revision=args.revision)
    resolved = str(info.sha)
    paths = list_repo_files(REPO, repo_type="dataset", revision=resolved)
    if args.list:
        print(f"# {REPO}@{resolved}")
        for path in paths:
            print(path)
        return
    if not args.pattern:
        raise SystemExit("Specify --pattern or --list. The dataset is large, so implicit full download is disabled.")
    selected = [x for x in paths if fnmatch.fnmatch(x, args.pattern)]
    if not selected:
        raise SystemExit(f"No files matched {args.pattern!r} at revision {resolved}")
    args.dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(selected)} files from {REPO}@{resolved}")
    downloaded: list[tuple[str, Path]] = []
    for i, path in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {path}")
        local = Path(hf_hub_download(REPO, path, repo_type="dataset", revision=resolved, local_dir=args.dest))
        downloaded.append((path, local))
    manifest = build_download_manifest(
        requested_revision=args.revision,
        resolved_revision=resolved,
        pattern=args.pattern,
        files=downloaded,
    )
    manifest_path = args.manifest or args.dest / "rail_vivid_download_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote provenance manifest {manifest_path} selection_sha256={manifest['selection_sha256']}")


if __name__ == "__main__":
    main()
