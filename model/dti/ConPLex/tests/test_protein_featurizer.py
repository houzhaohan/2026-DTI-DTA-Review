import os
import types
from pathlib import Path
import builtins

import torch

from conplex_dti.featurizer.protein import ProtBertFeaturizer


def test_protbert_featurizer_falls_back_when_transformers_is_unavailable(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "transformers":
            raise OSError("network unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    featurizer = ProtBertFeaturizer(save_dir=Path("/tmp"))
    result = featurizer("ACDE")

    assert result.shape == torch.Size([1024])


def test_protbert_featurizer_uses_hf_mirror_by_default(monkeypatch, tmp_path):
    real_import = builtins.__import__

    class DummyModel:
        pass

    class DummyTokenizer:
        pass

    class DummyPipeline:
        def __init__(self, *args, **kwargs):
            pass

    fake_transformers = types.SimpleNamespace(
        AutoModel=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: DummyModel()),
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: DummyTokenizer()),
        pipeline=lambda *args, **kwargs: DummyPipeline(),
    )

    def fake_import(name, *args, **kwargs):
        if name == "transformers":
            return fake_transformers
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)

    ProtBertFeaturizer(save_dir=tmp_path)

    assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"
