from pathlib import Path
import math

from ml.prepare_rail_vivid import anomaly_label, nearest_image, alignment_quality_summary
import numpy as np


def test_anomaly_is_unknown_without_ground_truth_catalog():
    label, anomaly_id, distance = anomaly_label(40.0, -77.0, [])
    assert math.isnan(label)
    assert anomaly_id is None and distance is None


def test_anomaly_catalog_labels_by_geospatial_tolerance():
    catalog = [{"id": "A1", "latitude": 40.0, "longitude": -77.0, "radius_m": 20.0}]
    positive, anomaly_id, distance = anomaly_label(40.00005, -77.0, catalog)
    negative, _, _ = anomaly_label(40.001, -77.0, catalog)
    assert positive == 1.0 and anomaly_id == "A1" and distance < 20.0
    assert negative == 0.0


def test_nearest_image_reports_alignment_error():
    times = np.array([10.0, 10.1, 10.2])
    paths = [Path("a.jpg"), Path("b.jpg"), Path("c.jpg")]
    path, delta = nearest_image(10.14, times, paths)
    assert path.endswith("b.jpg")
    assert abs(delta - 0.04) < 1e-9


def test_alignment_qa_summary_exposes_tail_error_and_acceptance_rate():
    qa=alignment_quality_summary([1.0,2.0,10.0,20.0],total_bins=5,rejected_far_image=1)
    assert qa["accepted_windows"]==4 and qa["rejected_far_image"]==1
    assert abs(qa["acceptance_rate"]-0.8)<1e-9
    assert qa["image_time_error_ms"]["p50"]==6.0
    assert qa["image_time_error_ms"]["max"]==20.0
