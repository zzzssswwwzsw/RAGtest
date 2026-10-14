"""settings.py —— 运行期配置：显存预算常量、档位预设与环境变量覆盖。

显存数字全部按 RTX 3060 Laptop 实测的 6144 MiB 规划，而非标称的 8GB。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

# ── 硬件基线（实测值，改动前请核对 nvidia-smi）──
VRAM_TOTAL_MIB = 6144

# ── 三重冗余的两根硬杠杆 ──
# MEM_FRACTION: 进程级硬上限比例，超限由 PyTorch 抛 OutOfMemoryError（可捕获、可降级），
#   而不是触发 WDDM 向共享内存换页导致整机卡死。
MEM_FRACTION = 0.62
# TOKEN_BUDGET: 组批上限 batch × length，规避注意力开销随序列长度平方增长。
TOKEN_BUDGET = 8192

# ── 显存需求估算分量（FP16 口径）──
# 权重：bge-m3 ≈ 568M 参数 × 2 字节
WEIGHTS_MIB = 1150
# CUDA 上下文 + cuBLAS/cuDNN 工作区，与 batch 无关的固定开销
CONTEXT_MIB = 500
# 体检时额外预留的余量，吸收碎片化与驱动侧未统计占用
SAFETY_MARGIN_MIB = 800

# CPU 兜底路径所需的最小可用系统内存
MIN_HOST_RAM_MIB = 4096

MODEL_ID = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# 显式落盘的权重复本：存在就直接用，省掉每次启动的联网解析，也让服务能离线起。
LOCAL_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "bge-m3"


def resolve_model_id() -> str:
    """按 环境变量 > 本地权重目录 > Hub 仓库名 的顺序决定模型来源。"""
    override = os.environ.get("BGE_MODEL_ID", "").strip()
    if override:
        return override
    if LOCAL_MODEL_DIR.is_dir():
        return str(LOCAL_MODEL_DIR)
    return MODEL_ID


@dataclass(frozen=True)
class Preset:
    """一个显存档位：序列长度、批大小与其对应的激活显存预估。"""

    name: str
    max_length: int
    batch: int
    activation_mib: int

    @property
    def key(self) -> str:
        """可回写进环境变量的档位标识。

        name 会被加上 (override)/(forced) 之类的人类可读后缀，那个后缀不是
        PRESETS 的键，直接写进 .env.runtime 会让下次 load_settings 报未知档位。
        """
        return self.name.split("(", 1)[0].strip()


# activation_mib 是 FP16 前向的"激活"峰值（不含权重），由 scripts/bench.py 在本机实测
# 得出后乘 1.2 余量回填（实测于 RTX 3060 Laptop 6GB / 2026-10-14）：
#   512×8 峰值 1277 MiB、1024×8 峰值 1717 MiB、4096×1 峰值 2154 MiB（均含 1093 MiB 权重）。
# 换机器或换 torch 版本后应重跑 bench.py 再回填。
PRESETS: dict[str, Preset] = {
    "safe": Preset("safe", max_length=512, batch=8, activation_mib=250),
    "balanced": Preset("balanced", max_length=1024, batch=8, activation_mib=800),
    # long 取 4096 而非 8192：实测 8192 单条在本卡上必然 OOM，
    # 且会静默退化到 CPU（单条约 160s），故不作为可用档位。
    "long": Preset("long", max_length=4096, batch=1, activation_mib=1300),
}

DEFAULT_PRESET = "safe"


@dataclass(frozen=True)
class Settings:
    """一次进程生命周期内生效的配置快照。"""

    preset: Preset
    mem_fraction: float
    token_budget: int
    model_id: str
    backend: str  # auto | flagembedding | transformers
    device: str  # cuda | cpu
    concurrency: int
    force: bool  # 跳过体检阻塞，自动降档继续

    @property
    def vram_limit_mib(self) -> int:
        """进程级硬上限对应的 MiB 值，用于日志与 /healthz 展示。"""
        return int(VRAM_TOTAL_MIB * self.mem_fraction)

    @property
    def need_mib(self) -> int:
        """加载模型所需的最小空闲显存 = 权重 + 上下文 + 激活预算 + 安全余量。"""
        return (
            WEIGHTS_MIB
            + CONTEXT_MIB
            + self.preset.activation_mib
            + SAFETY_MARGIN_MIB
        )


# ── 环境变量读取（非法值直接报错，避免带病启动）──


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"环境变量 {name} 需为整数，收到 {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"环境变量 {name} 需为正整数，收到 {value}")
    return value


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"环境变量 {name} 需为浮点数，收到 {raw!r}") from exc
    if not 0.0 < value <= 1.0:
        raise ValueError(f"环境变量 {name} 需落在 (0, 1]，收到 {value}")
    return value


def load_settings(force: bool | None = None) -> Settings:
    """按 环境变量 > 默认值 的顺序装配配置；force 为 None 时读 BGE_FORCE。"""
    preset_name = _env_str("BGE_PRESET", DEFAULT_PRESET)
    if preset_name not in PRESETS:
        raise ValueError(
            f"未知档位 {preset_name!r}，可选：{', '.join(PRESETS)}"
        )
    preset = PRESETS[preset_name]

    # 单独覆盖长度/批量时，档位名不再准确，故显式改写而非静默沿用
    override_length = _env_int("BGE_MAX_LENGTH", 0)
    override_batch = _env_int("BGE_BATCH", 0)
    if override_length or override_batch:
        preset = replace(
            preset,
            name=f"{preset.name}(override)",
            max_length=override_length or preset.max_length,
            batch=override_batch or preset.batch,
        )

    backend = _env_str("BGE_BACKEND", "auto").lower()
    if backend not in ("auto", "flagembedding", "transformers"):
        raise ValueError(
            f"BGE_BACKEND 需为 auto/flagembedding/transformers，收到 {backend!r}"
        )

    device = _env_str("BGE_DEVICE", "cuda").lower()
    if device not in ("cuda", "cpu"):
        raise ValueError(f"BGE_DEVICE 需为 cuda/cpu，收到 {device!r}")

    if force is None:
        force = os.environ.get("BGE_FORCE", "").strip().lower() in ("1", "true", "yes")

    return Settings(
        preset=preset,
        mem_fraction=_env_float("BGE_MEM_FRACTION", MEM_FRACTION),
        token_budget=_env_int("BGE_TOKEN_BUDGET", TOKEN_BUDGET),
        model_id=resolve_model_id(),
        backend=backend,
        device=device,
        concurrency=_env_int("BGE_CONCURRENCY", 2),
        force=force,
    )


def apply_memory_fraction(fraction: float) -> None:
    """把进程级显存硬上限交给 PyTorch。

    必须在 CUDA 初始化（首次分配）之前调用，否则限制只对后续分配生效。
    调用方需保证 torch 已导入；此处只做延迟导入，避免 settings 依赖重型包。
    """
    import torch  # noqa: PLC0415 —— 延迟导入，纯配置场景不应拉起 torch

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)
