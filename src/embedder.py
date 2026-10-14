"""embedder.py —— bge-m3 稠密向量编码器：FP16 加载、token 预算组批、OOM 自愈。

对外只产出 1024 维 dense 向量（L2 已归一化），不启用 sparse / ColBERT 多向量。
"""

from __future__ import annotations

import logging
import threading

import numpy as np
import torch

from .settings import EMBEDDING_DIM, Settings, apply_memory_fraction

log = logging.getLogger("bge.embedder")

# torch 2.3 没有 torch.OutOfMemoryError（2.5 才补上这个别名），只有 cuda 下的这一个。
# 捕获它必须在 except 处求值，写错名字会让降级路径自己抛 AttributeError。
CUDA_OOM = torch.cuda.OutOfMemoryError


# ── 组批 ──


def pack_batches(
    lengths: list[int],
    max_batch: int,
    token_budget: int,
) -> list[list[int]]:
    """按「批大小 × 批内最长序列 ≤ token_budget」装箱，返回每批的下标列表。

    先长后短的贪心装箱：长文本先落位，短文本填缝，批数更少且每批 padding 浪费更小。
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    batches: list[list[int]] = []
    current: list[int] = []
    for idx in order:
        if current:
            longest = max(lengths[current[0]], lengths[idx])
            projected = longest * (len(current) + 1)
            if len(current) + 1 > max_batch or projected > token_budget:
                batches.append(current)
                current = [idx]
                continue
        current.append(idx)
    if current:
        batches.append(current)
    # 批内按下标升序还原原文顺序，让 padding 长度分布更平稳
    return [sorted(batch) for batch in batches]


# ── 后端 ──


class _FlagEmbeddingBackend:
    """官方封装路径：pooling / 归一化/ fp16 细节由 FlagEmbedding 负责。"""

    name = "flagembedding"

    def __init__(self, model_id: str, device: str) -> None:
        from FlagEmbedding import BGEM3FlagModel  # noqa: PLC0415 —— 重型依赖延迟导入

        self.model = BGEM3FlagModel(model_id, use_fp16=(device == "cuda"), device=device)

    def encode(self, texts: list[str], max_length: int) -> np.ndarray:
        output = self.model.encode(
            texts,
            batch_size=len(texts),
            max_length=max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return np.asarray(output["dense_vecs"], dtype=np.float32)

    def count_tokens(self, texts: list[str], max_length: int) -> int:
        encoded = self.model.tokenizer(texts, truncation=True, max_length=max_length)
        return sum(len(ids) for ids in encoded["input_ids"])


class _TransformersBackend:
    """等价实现（方案第八节回退路径）：XLM-R + CLS pooling + L2 归一化。"""

    name = "transformers"

    def __init__(self, model_id: str, device: str) -> None:
        from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = (
            AutoModel.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            .to(device)
            .eval()
        )

    def encode(self, texts: list[str], max_length: int) -> np.ndarray:
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}
        with torch.inference_mode():
            hidden = self.model(**batch).last_hidden_state
        # bge-m3 的 dense 向量取 CLS 位后做 L2 归一化
        cls = hidden[:, 0, :]
        return torch.nn.functional.normalize(cls.float(), p=2, dim=1).cpu().numpy()

    def count_tokens(self, texts: list[str], max_length: int) -> int:
        encoded = self.tokenizer(texts, truncation=True, max_length=max_length)
        return sum(len(ids) for ids in encoded["input_ids"])


def _build_backend(settings: Settings):
    """按 backend 配置挑后端；auto 表示优先 FlagEmbedding，缺库则退回 transformers。"""
    if settings.backend == "flagembedding":
        return _FlagEmbeddingBackend(settings.model_id, settings.device)
    if settings.backend == "transformers":
        return _TransformersBackend(settings.model_id, settings.device)
    try:
        return _FlagEmbeddingBackend(settings.model_id, settings.device)
    except ImportError as exc:
        log.warning("FlagEmbedding 不可用（%s），回退到 transformers 直调", exc)
        return _TransformersBackend(settings.model_id, settings.device)


# ── 编码器 ──


class Embedder:
    """进程内单例式编码器：加载一次模型，长期驻留显存。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._backend = None
        self._cpu_backend = None
        self._sem = threading.Semaphore(settings.concurrency)
        self._degraded_events: list[str] = []

    # ── 生命周期 ──

    def load(self) -> None:
        """加载模型。显存硬上限必须在任何 CUDA 分配之前设好。"""
        if self.settings.device == "cuda":
            apply_memory_fraction(self.settings.mem_fraction)
            log.info(
                "显存硬上限已设为 %d MiB（总量 %d × %.2f）",
                self.settings.vram_limit_mib,
                self.settings.vram_limit_mib / self.settings.mem_fraction,
                self.settings.mem_fraction,
            )
        self._backend = _build_backend(self.settings)
        log.info(
            "模型 %s 已加载（后端 %s，设备 %s，档位 %s）",
            self.settings.model_id,
            self._backend.name,
            self.settings.device,
            self.settings.preset.name,
        )

    @property
    def backend_name(self) -> str:
        return self._backend.name if self._backend else "unloaded"

    @property
    def loaded(self) -> bool:
        return self._backend is not None

    @property
    def degradations(self) -> list[str]:
        """累计的降级事件，供 /healthz 暴露运行期健康度。"""
        return list(self._degraded_events)

    # ── 编码 ──

    def encode(self, texts: list[str], max_length: int | None = None) -> np.ndarray:
        """编码一批文本，返回 [n, 1024] 的 L2 归一化向量。"""
        if self._backend is None:
            raise RuntimeError("模型尚未加载，请先调用 load()")
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        limit = min(max_length or self.settings.preset.max_length, self.settings.preset.max_length)
        vectors = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
        lengths = [self._token_length(text, limit) for text in texts]

        with self._sem:
            for batch in pack_batches(lengths, self.settings.preset.batch, self.settings.token_budget):
                chunk = [texts[i] for i in batch]
                result = self._encode_with_fallback(chunk, limit)
                for row, original_index in enumerate(batch):
                    vectors[original_index] = result[row]
        return vectors

    def count_tokens(self, texts: list[str], max_length: int | None = None) -> int:
        """按实际分词结果统计 token 数（已计入截断），用于 usage 上报。"""
        if self._backend is None:
            raise RuntimeError("模型尚未加载，请先调用 load()")
        limit = min(max_length or self.settings.preset.max_length, self.settings.preset.max_length)
        return self._backend.count_tokens(texts, limit)

    def _token_length(self, text: str, limit: int) -> int:
        """估长只用于组批，取"字符数"这一上界即可，避免真实分词的开销。"""
        return max(1, min(len(text), limit))

    def _encode_with_fallback(self, texts: list[str], limit: int) -> np.ndarray:
        """OOM 时逐级降级：清缓存 → 减半重试 → 单条转 CPU，服务本身不崩。"""
        try:
            return self._backend.encode(texts, limit)
        except CUDA_OOM:
            self._on_oom(texts, "首次前向 OOM")
            torch.cuda.empty_cache()

        if len(texts) > 1:
            mid = len(texts) // 2
            head = self._encode_with_fallback(texts[:mid], limit)
            tail = self._encode_with_fallback(texts[mid:], limit)
            return np.concatenate([head, tail], axis=0)

        # 单条仍 OOM：降到 CPU 兜底，保证请求有结果而不是 500
        self._remember(
            f"文本 {len(texts[0])} 字符（截断后 ≤ {limit} token）的样本已转 CPU 兜底处理"
        )
        return self._cpu_backend_for().encode(texts, limit)

    def _on_oom(self, texts: list[str], stage: str) -> None:
        allocated = torch.cuda.memory_allocated() // (1024 * 1024)
        log.warning(
            "%s：批内 %d 条、最长 %d 字，当前已分配 %d MiB，开始回收并降级",
            stage,
            len(texts),
            max(len(t) for t in texts),
            allocated,
        )

    def _remember(self, message: str) -> None:
        self._degraded_events.append(message)
        log.warning("%s", message)

    def _cpu_backend_for(self):
        """CPU 兜底编码器（fp32），首次用到时才加载，避免白白占用系统内存。"""
        if self._cpu_backend is None:
            from dataclasses import replace  # noqa: PLC0415

            cpu_settings = replace(self.settings, device="cpu", backend="transformers")
            self._cpu_backend = _build_backend(cpu_settings)
            log.warning("CPU 兜底编码器已加载（fp32）")
        return self._cpu_backend

    # ── 运行期度量 ──

    def memory_stats(self) -> dict:
        """当前显存占用快照，供 /healthz 展示。"""
        if self.settings.device != "cuda":
            return {"device": "cpu"}
        mib = 1024 * 1024
        return {
            "device": "cuda",
            "limit_mib": self.settings.vram_limit_mib,
            "allocated_mib": torch.cuda.memory_allocated() // mib,
            "reserved_mib": torch.cuda.memory_reserved() // mib,
            "peak_mib": torch.cuda.max_memory_allocated() // mib,
        }
