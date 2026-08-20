from __future__ import annotations

from ml.evaluate_multimodal import markdown, summarize_group_variability


def test_markdown_distinguishes_untouched_test_from_validation_and_ablation_semantics():
    metric = {
        'vibration_mae': [1, 2, 3],
        'vibration_rmse': [1, 2, 3],
        'vision_mae': [.1, .2, .3],
        'vision_rmse': [.1, .2, .3],
        'vibration_mae_mean': 2.0,
        'vision_mae_mean': .2,
    }
    report = {
        'split_mode':'run',
        'test_groups':['run-test'],
        'validation_groups':['run-val'],
        'test_group_count':1,
        'validation_group_count':1,
        'windows': 17,
        'sample_period_ms': 100.0,
        'metrics': {k: dict(metric) for k in [
            'persistence', 'nonvisual_inputs_ablation', 'visual_inputs_ablation', 'multimodal'
        ]},
    }
    text = markdown(report)
    assert 'run-test' in text and 'run-val' in text
    assert 'nonvisual_inputs_ablation' in text and 'visual_inputs_ablation' in text
    assert 'not the performance of separately trained unimodal models' in text


def test_group_variability_reports_between_location_distribution():
    def m(value):
        return {"vibration_mae_mean": value}
    groups = {
        "A": {"multimodal": m(1.0), "persistence": m(2.0)},
        "B": {"multimodal": m(2.0), "persistence": m(2.0)},
        "C": {"multimodal": m(3.0), "persistence": m(4.0)},
    }
    summary = summarize_group_variability(groups)
    assert summary["groups_with_windows"] == 3
    assert summary["multimodal_vibration_mae_mean"]["median"] == 2.0
    assert summary["relative_vibration_mae_improvement"]["median"] == 0.25


def test_markdown_includes_group_variability_when_available():
    metric = {
        'vibration_mae': [1, 2, 3], 'vibration_rmse': [1, 2, 3],
        'vision_mae': [.1, .2, .3], 'vision_rmse': [.1, .2, .3],
        'vibration_mae_mean': 2.0, 'vision_mae_mean': .2,
    }
    report = {
        'split_mode':'spatial','test_groups':['A'],'validation_groups':['B'],
        'test_group_count':1,'validation_group_count':1,'windows':10,'sample_period_ms':100.0,
        'metrics': {k: dict(metric) for k in ['persistence','nonvisual_inputs_ablation','visual_inputs_ablation','multimodal']},
        'group_variability': {
            'per_group': {'A': {'windows':10,'persistence':dict(metric),'multimodal':dict(metric)}},
            'summary': summarize_group_variability({'A': {'persistence':dict(metric),'multimodal':dict(metric)}}),
        },
    }
    text=markdown(report)
    assert 'Held-out group variability' in text
    assert '| A | 10 |' in text
