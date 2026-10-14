"""api.py —— FastAPI 服务：OpenAI 兼容 /v1/embeddings，附带 /healthz 与 /preflight。

必须以单 worker 启动：多 worker 会在显存中复制多份模型，6GB 卡直接爆。
"""

from __future__ import annotations

import base64
import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .embedder import CUDA_OOM, Embedder
from .preflight import diagnose, ensure_vram, harden_console
from .settings import EMBEDDING_DIM, Settings, load_settings

log = logging.getLogger("bge.api")

# 单请求最多接受的文本条数：显存受限场景下，宁可让调用方分片也不要长队列堆积
MAX_INPUT_ITEMS = 256


# ── 请求 / 响应模型 ──


class EmbeddingRequest(BaseModel):
    """OpenAI /v1/embeddings 的入参子集。"""

    input: str | list[str]
    model: str = "bge-m3"
    encoding_format: str = "float"
    max_length: int | None = Field(default=None, ge=1)
    dimensions: int | None = Field(default=None, ge=1)


class EmbeddingItem(BaseModel):
    object: str = "embedding"
    index: int
    embedding: list[float] | str


class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingItem]
    model: str
    usage: Usage


# ── 运行时状态 ──


class _State:
    """单进程内的模型与配置快照。"""

    settings: Settings | None = None
    embedder: Embedder | None = None


state = _State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    harden_console()
    # uvicorn 只给自身 logger 配了 handler，不配根 logger 的话本服务的 INFO 诊断
    # （显存硬上限、档位、后端）会被全部丢弃，只剩 WARNING 走 last-resort 输出。
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    # 服务模式跳过交互阻塞：显存不足时打印警告并自动降档
    gated = ensure_vram(settings, interactive=False)
    if gated is None:
        raise RuntimeError("显存体检未通过，服务拒绝启动")
    embedder = Embedder(gated)
    embedder.load()
    state.settings = gated
    state.embedder = embedder
    # 端口由启动方决定，日志里不写死 URL，避免与实际监听端口不一致
    log.info("服务就绪，接口见 /docs、/healthz、/preflight")
    try:
        yield
    finally:
        state.embedder = None
        state.settings = None


app = FastAPI(
    title="bge-m3 embedding service",
    description="BGE-M3 稠密向量（1024 维）本地推理服务，OpenAI 兼容接口",
    version="1.0.0",
    lifespan=lifespan,
)


def _require_state() -> tuple[Settings, Embedder]:
    """配置与编码器同生共死，任一缺失都说明模型还没加载好。"""
    if state.embedder is None or state.settings is None:
        raise HTTPException(status_code=503, detail="模型尚未就绪")
    return state.settings, state.embedder


# ── 接口 ──


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
def create_embeddings(payload: EmbeddingRequest) -> EmbeddingResponse:
    """把文本编码为 L2 归一化的 1024 维向量。"""
    settings, embedder = _require_state()

    texts = [payload.input] if isinstance(payload.input, str) else list(payload.input)
    if not texts or any(not isinstance(t, str) for t in texts):
        raise HTTPException(status_code=400, detail="input 不能为空，且每项必须是字符串")
    if len(texts) > MAX_INPUT_ITEMS:
        raise HTTPException(
            status_code=400,
            detail=f"单次请求最多 {MAX_INPUT_ITEMS} 条文本，收到 {len(texts)} 条",
        )
    if payload.encoding_format not in ("float", "base64"):
        raise HTTPException(
            status_code=400,
            detail=f"不支持的 encoding_format：{payload.encoding_format}",
        )
    if payload.dimensions not in (None, EMBEDDING_DIM):
        raise HTTPException(
            status_code=400,
            detail=f"bge-m3 固定输出 {EMBEDDING_DIM} 维，不支持 dimensions={payload.dimensions}",
        )
    if payload.max_length and payload.max_length > settings.preset.max_length:
        raise HTTPException(
            status_code=400,
            detail=(
                f"max_length 上限为当前档位的 {settings.preset.max_length}，"
                f"收到 {payload.max_length}"
            ),
        )

    try:
        vectors = embedder.encode(texts, payload.max_length)
        tokens = embedder.count_tokens(texts, payload.max_length)
    except CUDA_OOM as exc:
        # 逐级降级仍失败：显存类问题用 507 表达，区别于 500 的服务端缺陷
        log.error("显存耗尽且降级失败：%s", exc)
        raise HTTPException(status_code=507, detail="显存不足，请缩短文本或降低并发") from exc

    data = [
        EmbeddingItem(index=i, embedding=_format_vector(vectors[i], payload.encoding_format))
        for i in range(len(texts))
    ]
    return EmbeddingResponse(
        data=data,
        model=payload.model,
        usage=Usage(prompt_tokens=tokens, total_tokens=tokens),
    )


def _format_vector(vector: np.ndarray, encoding_format: str) -> list[float] | str:
    if encoding_format == "base64":
        return base64.b64encode(np.asarray(vector, dtype=np.float32).tobytes()).decode("ascii")
    return [float(x) for x in vector]


@app.get("/healthz")
def healthz() -> dict:
    """存活探针：附带当前档位、后端与显存占用。"""
    settings, embedder = _require_state()
    return {
        "status": "ok",
        "model": settings.model_id,
        "backend": embedder.backend_name,
        "preset": settings.preset.name,
        "max_length": settings.preset.max_length,
        "batch": settings.preset.batch,
        "token_budget": settings.token_budget,
        "memory": embedder.memory_stats(),
        "degradations": embedder.degradations,
    }


@app.get("/preflight")
def preflight() -> dict:
    """运行期显存体检：随时可查当前还有多少冗余。"""
    # 模型尚未加载时也能回答，用默认配置估算需求即可
    settings = state.settings or load_settings()
    return diagnose(settings).to_dict()
