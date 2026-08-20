import pytest

from ml.export_onnx import validate_deployment_columns, validate_model_architecture, build_deployment_manifest
from ml.railguard_ml.contracts import DEPLOYMENT_IMAGE_MODE, DEPLOYMENT_SENSOR_COLUMNS, MODEL_ARCH_VERSION
from ml.train_multimodal import SENSOR_COLUMNS


def test_training_and_export_share_native_feature_contract():
    assert SENSOR_COLUMNS == DEPLOYMENT_SENSOR_COLUMNS
    validate_deployment_columns(list(DEPLOYMENT_SENSOR_COLUMNS))


def test_export_rejects_reordered_or_extended_features():
    reordered = list(DEPLOYMENT_SENSOR_COLUMNS)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(ValueError):
        validate_deployment_columns(reordered)
    with pytest.raises(ValueError):
        validate_deployment_columns(list(DEPLOYMENT_SENSOR_COLUMNS) + ["vision_sharpness"])


def test_export_rejects_checkpoint_from_implicit_or_old_architecture():
    with pytest.raises(ValueError):
        validate_model_architecture({})
    with pytest.raises(ValueError):
        validate_model_architecture({"model_arch_version": MODEL_ARCH_VERSION - 1})
    validate_model_architecture({"model_arch_version": MODEL_ARCH_VERSION})


def test_deployment_manifest_binds_checkpoint_and_onnx(tmp_path):
    checkpoint=tmp_path/'model.pt'; onnx=tmp_path/'model.onnx'
    checkpoint.write_bytes(b'checkpoint-bytes'); onnx.write_bytes(b'onnx-bytes')
    ckpt={
        'model_arch_version':MODEL_ARCH_VERSION,
        'sensor_columns':list(DEPLOYMENT_SENSOR_COLUMNS),'image_mode':DEPLOYMENT_IMAGE_MODE,'seq_len':32,'sample_period_s':0.1,
        'split_mode':'spatial','spatial_block_m':500.0,'spatial_purge_margin_m':30.0,'spatial_purged_train_rows':12,'spatial_purged_validation_rows':4,'train_groups':['a'],'validation_groups':['b'],'test_groups':['c'],
        'provenance':{'data_sha256':'abc'}
    }
    m=build_deployment_manifest(checkpoint,onnx,ckpt)
    assert m['model_version'].startswith('fusion-')
    assert len(m['checkpoint_sha256'])==64 and len(m['onnx_sha256'])==64
    assert m['sample_period_ms']==100.0
    assert m['sensor_columns']==DEPLOYMENT_SENSOR_COLUMNS
    assert m['image_mode']==DEPLOYMENT_IMAGE_MODE
    assert m['model_arch_version']==MODEL_ARCH_VERSION
    assert m['spatial_purge_margin_m']==30.0
    assert m['spatial_purged_train_rows']==12
