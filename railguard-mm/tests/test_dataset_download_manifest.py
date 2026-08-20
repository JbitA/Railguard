from pathlib import Path

from ml.download_rail_vivid import build_download_manifest


def test_download_manifest_binds_revision_paths_sizes_and_bytes(tmp_path: Path):
    a = tmp_path / "a.csv"; b = tmp_path / "frame.jpg"
    a.write_bytes(b"one"); b.write_bytes(b"two")
    first = build_download_manifest(
        requested_revision="main",
        resolved_revision="abc123",
        pattern="*",
        files=[("frame.jpg", b), ("a.csv", a)],
    )
    assert first["resolved_revision"] == "abc123"
    assert [x["repo_path"] for x in first["files"]] == ["a.csv", "frame.jpg"]
    before = first["selection_sha256"]
    b.write_bytes(b"changed")
    second = build_download_manifest(
        requested_revision="main",
        resolved_revision="abc123",
        pattern="*",
        files=[("a.csv", a), ("frame.jpg", b)],
    )
    assert second["selection_sha256"] != before
