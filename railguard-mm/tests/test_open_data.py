from pathlib import Path

from PIL import Image

from ml.open_data import load_catalog, import_surface_faults, verify_surface_faults_layout


CLASSES = ("Cracks", "Flakings", "Grooves", "Joints", "Shellings", "Spallings", "Squats")


def _make_surface_tree(root: Path, images_per_class: int = 2) -> None:
    for cls in CLASSES:
        folder = root / cls
        folder.mkdir(parents=True)
        for i in range(images_per_class):
            Image.new("L", (8, 8), color=20 + i).save(folder / f"{i}.png")


def test_open_data_catalog_contains_only_attributed_sources():
    sources = load_catalog()["sources"]
    assert sources["rail_vivid"]["license"] == "CC BY 4.0"
    assert sources["railway_surface_faults"]["license"] == "CC BY 4.0"
    assert sources["rail_vivid"]["doi"] == "10.57967/hf/8411"
    assert sources["railway_surface_faults"]["doi"] == "10.17632/8hxtgyyxrw.2"


def test_surface_fault_import_writes_content_manifest(tmp_path: Path):
    source = tmp_path / "provider"
    dest = tmp_path / "imported"
    _make_surface_tree(source)
    layout = verify_surface_faults_layout(source)
    assert set(layout) == set(CLASSES)
    manifest_path = import_surface_faults(source, dest, copy=False)
    text = manifest_path.read_text()
    assert '"image_count": 14' in text
    assert '"selection_sha256"' in text
    assert '"license": "CC BY 4.0"' in text
