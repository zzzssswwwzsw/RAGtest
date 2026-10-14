"""test_smoke.py —— 冒烟测试：组批与配置的纯逻辑 + 需要真实模型的服务链路。

模型相关用例默认跳过（要下 2.3GB 权重、吃 3GB 显存），
需要时设 BGE_SMOKE_MODEL=1 打开：
    BGE_SMOKE_MODEL=1 python -m pytest tests/ -v
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from src.embedder import pack_batches
from src.preflight import degrade, diagnose, query_vram
from src.settings import PRESETS, load_settings

NEEDS_MODEL = pytest.mark.skipif(
    os.environ.get("BGE_SMOKE_MODEL", "") != "1",
    reason="需要真实模型权重与显存，设 BGE_SMOKE_MODEL=1 开启",
)


# ── 组批逻辑 ──


def test_pack_batches_respects_token_budget():
    """8 条 512 字的文本在 8192 token 预算下应合成一批。"""
    batches = pack_batches([512] * 8, max_batch=8, token_budget=8192)
    assert batches == [list(range(8))]


def test_pack_batches_splits_when_exceeding_budget():
    """2048 字 × 8 条远超 8192 预算，按 4 条一批切开。"""
    batches = pack_batches([2048] * 8, max_batch=8, token_budget=8192)
    assert len(batches) == 2
    assert all(len(b) == 4 for b in batches)


def test_pack_batches_puts_long_text_alone():
    """单条就要吃满预算的长文本，必须独占一批。"""
    batches = pack_batches([8192, 10, 10], max_batch=8, token_budget=8192)
    alone = [b for b in batches if 0 in b]
    assert alone == [[0]]


def test_pack_batches_covers_all_indices_exactly_once():
    lengths = [5, 900, 120, 7000, 33, 512, 2048, 64]
    batches = pack_batches(lengths, max_batch=4, token_budget=4096)
    assert sorted(i for b in batches for i in b) == sorted(range(len(lengths)))
    for batch in batches:
        assert len(batch) <= 4
        # 单条就超预算的文本独占一批：它随后被分词器截断到 max_length，
        # 只要不被塞进多条的批里，预算约束在语义上仍然成立。
        if len(batch) > 1:
            assert max(lengths[i] for i in batch) * len(batch) <= 4096


# ── 配置 ──


def test_defaults_match_plan():
    settings = load_settings()
    if not os.environ.get("BGE_PRESET"):
        assert settings.preset.name == "safe"
        assert settings.preset.max_length == 512
        assert settings.preset.batch == 8
    assert settings.vram_limit_mib == int(6144 * 0.62)


def test_env_override_applies(monkeypatch):
    monkeypatch.setenv("BGE_PRESET", "long")
    monkeypatch.setenv("BGE_BATCH", "3")
    monkeypatch.setenv("BGE_MEM_FRACTION", "0.5")
    settings = load_settings()
    assert settings.preset.name.startswith("long")
    assert settings.preset.batch == 3
    assert settings.preset.max_length == PRESETS["long"].max_length
    assert settings.mem_fraction == 0.5


def test_invalid_env_rejected(monkeypatch):
    monkeypatch.setenv("BGE_PRESET", "turbo")
    with pytest.raises(ValueError, match="未知档位"):
        load_settings()
    monkeypatch.delenv("BGE_PRESET")

    monkeypatch.setenv("BGE_MEM_FRACTION", "1.5")
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        load_settings()


def test_preset_key_strips_display_suffix(monkeypatch):
    """档位名带 (override) 展示后缀时，key 必须退化成合法档位名。

    回归点：preflight 曾把 BGE_PRESET=safe(forced) 写进 .env.runtime，
    服务下次启动因为 "未知档位" 直接拒绝运行。
    """
    monkeypatch.setenv("BGE_PRESET", "safe")
    monkeypatch.setenv("BGE_BATCH", "1")
    overridden = load_settings()
    assert overridden.preset.name == "safe(override)"
    assert overridden.preset.key == "safe"


def test_degraded_preset_round_trips_through_env(monkeypatch):
    """降档后的档位回写环境变量再读回来，必须仍然合法且数值不丢。"""
    monkeypatch.setenv("BGE_PRESET", "balanced")
    monkeypatch.delenv("BGE_MAX_LENGTH", raising=False)
    monkeypatch.delenv("BGE_BATCH", raising=False)
    degraded = degrade(load_settings())
    assert degraded.preset.name == "safe(forced)"

    monkeypatch.setenv("BGE_PRESET", degraded.preset.key)
    monkeypatch.setenv("BGE_MAX_LENGTH", str(degraded.preset.max_length))
    monkeypatch.setenv("BGE_BATCH", str(degraded.preset.batch))
    restored = load_settings()
    assert restored.preset.key == "safe"
    assert restored.preset.max_length == 512
    assert restored.preset.batch == 1


def test_need_scales_with_preset(monkeypatch):
    monkeypatch.setenv("BGE_PRESET", "safe")
    safe_need = load_settings().need_mib
    monkeypatch.setenv("BGE_PRESET", "long")
    assert load_settings().need_mib > safe_need


# ── 体检 ──


def test_diagnose_reports_vram():
    """真实读一次显存：拿不到也必须是可解释的结果，而不是异常。"""
    report = query_vram()
    assert report.source in ("nvml", "nvidia-smi", "unknown")
    if report.known:
        assert report.used_mib + report.free_mib <= report.total_mib + 64  # 允许驱动统计抖动

    diag = diagnose(load_settings())
    payload = diag.to_dict()
    assert payload["need_mib"] > 0
    assert isinstance(payload["ok"], bool)


# ── 服务链路（需要模型）──


@NEEDS_MODEL
def test_embeddings_are_normalized_1024d():
    from fastapi.testclient import TestClient

    from src.api import app

    with TestClient(app) as client:
        response = client.post(
            "/v1/embeddings",
            json={"input": ["测试文本", "second sample"], "model": "bge-m3"},
        )
        assert response.status_code == 200
        body = response.json()
        assert len(body["data"]) == 2
        assert body["usage"]["prompt_tokens"] > 0

        for item in body["data"]:
            vector = np.asarray(item["embedding"], dtype=np.float32)
            assert vector.shape == (1024,)
            assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-3


@NEEDS_MODEL
def test_healthz_reports_preset():
    from fastapi.testclient import TestClient

    from src.api import app

    with TestClient(app) as client:
        body = client.get("/healthz").json()
        assert body["status"] == "ok"
        assert body["max_length"] == 512
        assert body["memory"]["peak_mib"] > 0


@NEEDS_MODEL
def test_rejects_oversized_batch():
    from fastapi.testclient import TestClient

    from src.api import MAX_INPUT_ITEMS, app

    with TestClient(app) as client:
        response = client.post(
            "/v1/embeddings",
            json={"input": ["x"] * (MAX_INPUT_ITEMS + 1)},
        )
        assert response.status_code == 400
