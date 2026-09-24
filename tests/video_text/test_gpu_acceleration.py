from pathlib import Path
import sys
import types

import pytest

import src.video_text.ppocrv5_backend as backend_module
from src.video_text.ppocrv5_backend import PPOCRv5MobileBackend


def test_gpu_backend_prefers_batched_onnxruntime_cuda(monkeypatch, tmp_path):
    model = tmp_path / 'inference.onnx'
    model.write_bytes(b'onnx')
    captured = {}

    class FakeORTPredictor:
        precision = 'fp32'
        def __init__(self, model_path, **kwargs):
            captured['model_path'] = Path(model_path)
            captured.update(kwargs)
        def predict(self, images):
            return []

    monkeypatch.setattr(backend_module.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(backend_module, 'default_ppocrv5_onnx_model', lambda: model)
    monkeypatch.setattr(backend_module, 'cuda_onnxruntime_available', lambda: True)
    monkeypatch.setattr(backend_module, 'ONNXRuntimeTextDetectionPredictor', FakeORTPredictor)

    backend = PPOCRv5MobileBackend(device='cuda', predictor=None)
    assert backend.gpu_engine == 'onnxruntime_cuda'
    assert backend.gpu_acceleration_error is None
    assert captured['model_path'] == model
    assert captured['thresh'] == .30
    assert captured['box_thresh'] == .50


def test_gpu_backend_can_request_tensorrt(monkeypatch, tmp_path):
    model = tmp_path / 'inference.onnx'
    model.write_bytes(b'onnx')
    captured = {}

    class FakeORTPredictor:
        precision = 'fp32'
        engine = 'onnxruntime_tensorrt'
        providers = ['TensorrtExecutionProvider','CUDAExecutionProvider','CPUExecutionProvider']
        def __init__(self, model_path, **kwargs):
            captured['model_path'] = Path(model_path)
            captured.update(kwargs)
        def predict(self, images):
            return []

    monkeypatch.setattr(backend_module.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(backend_module, 'default_ppocrv5_onnx_model', lambda: model)
    monkeypatch.setattr(backend_module, 'cuda_onnxruntime_available', lambda: True)
    monkeypatch.setattr(backend_module, 'ONNXRuntimeTextDetectionPredictor', FakeORTPredictor)

    backend = PPOCRv5MobileBackend(device='cuda', predictor=None, gpu_engine='tensorrt', batch_size=64)
    assert backend.gpu_engine == 'onnxruntime_tensorrt'
    assert captured['engine'] == 'tensorrt'
    assert captured['precision'] == 'fp32'
    assert captured['max_batch_size'] == 64


def test_gpu_backend_falls_back_to_paddle(monkeypatch, tmp_path):
    model = tmp_path / 'inference.onnx'
    model.write_bytes(b'onnx')
    captured = {}

    class BrokenORTPredictor:
        def __init__(self, *args, **kwargs):
            raise RuntimeError('boom')

    class FakeTextDetection:
        def __init__(self, **kwargs):
            captured.update(kwargs)
        def predict(self, images):
            return []

    monkeypatch.setattr(backend_module.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(backend_module, 'default_ppocrv5_onnx_model', lambda: model)
    monkeypatch.setattr(backend_module, 'cuda_onnxruntime_available', lambda: True)
    monkeypatch.setattr(backend_module, 'ONNXRuntimeTextDetectionPredictor', BrokenORTPredictor)
    monkeypatch.setitem(sys.modules, 'paddleocr', types.SimpleNamespace(TextDetection=FakeTextDetection))

    backend = PPOCRv5MobileBackend(device='cuda', predictor=None)
    assert backend.gpu_engine == 'paddle'
    assert 'boom' in backend.gpu_acceleration_error
    assert captured['device'] == 'gpu:0'
    assert captured['enable_hpi'] is False
