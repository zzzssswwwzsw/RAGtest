"""bench.py —— 逐档实测峰值显存与耗时，用真实数据校正 settings.PRESETS。

用法（仓库根目录）：
    python scripts/bench.py                # 默认阶梯 512/1024/2048/4096/8192
    python scripts/bench.py --lengths 512,1024
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

# scripts/ 不是包，直接运行时要手动把仓库根加进 import 路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.embedder import CUDA_OOM, Embedder  # noqa: E402
from src.preflight import harden_console, query_vram  # noqa: E402
from src.settings import TOKEN_BUDGET, WEIGHTS_MIB, Preset, load_settings  # noqa: E402

log = logging.getLogger("bge.bench")

DEFAULT_LADDER = (512, 1024, 2048, 4096, 8192)
RESULT_PATH = Path(".bench_result.json")

# 以常见汉字为词表：XLM-R 对中文大致一字符一 token，用来合成近似 token 数的文本
_VOCAB = "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严龙飞"

# 实测峰值相对理论预算的回退余量：留 20% 给碎片化与驱动侧开销
HEADROOM = 1.2


def synth_text(chars: int, seed: int) -> str:
    """合成指定字符数的文本，字符数≈token 数量级。"""
    rng = random.Random(seed)
    return "".join(rng.choice(_VOCAB) for _ in range(chars))


def run_rung(embedder: Embedder, length: int, batch: int) -> dict:
    """跑一档测量：返回耗时、PyTorch 峰值与设备侧显存增量。"""
    # 直接改写编码器的档位，避免每档重新加载模型；token_budget 放大以强制单批
    embedder.settings = replace(
        embedder.settings,
        preset=Preset(name=f"bench-{length}", max_length=length, batch=batch, activation_mib=0),
        token_budget=length * batch,
    )

    texts = [synth_text(length, seed=i) for i in range(batch)]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = query_vram()
    degradations_before = len(embedder.degradations)

    started = time.perf_counter()
    vectors = embedder.encode(texts)
    elapsed = time.perf_counter() - started

    # 编码器内部会在 OOM 后自己降到 CPU，这让本档"看起来成功"却根本没跑在 GPU 上。
    # 必须把它标出来，否则推荐表会把不可行档位判为通过。
    fell_back = embedder.degradations[degradations_before:]
    peak_mib = torch.cuda.max_memory_allocated() // (1024 * 1024)
    after = query_vram()
    return {
        "length": length,
        "batch": batch,
        "tokens": length * batch,
        "seconds": round(elapsed, 3),
        "seconds_per_text": round(elapsed / batch, 4),
        "torch_peak_mib": peak_mib,
        "device_used_delta_mib": max(0, after.used_mib - before.used_mib),
        "norm_ok": bool(abs(float((vectors[0] ** 2).sum()) ** 0.5 - 1.0) < 1e-3),
        "cpu_fallback": bool(fell_back),
        "degradations": fell_back,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="bge-m3 显存阶梯压测")
    parser.add_argument(
        "--lengths",
        default=",".join(str(x) for x in DEFAULT_LADDER),
        help="逗号分隔的序列长度阶梯",
    )
    parser.add_argument("--out", default=str(RESULT_PATH), help="结果落盘路径")
    parser.add_argument("--batch", type=int, default=0, help="强制指定批大小，覆盖按预算推导的值")
    args = parser.parse_args()

    harden_console()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]

    settings = load_settings()
    if settings.device != "cuda":
        print("未启用 CUDA，压测无意义", file=sys.stderr)
        return 2

    embedder = Embedder(settings)
    embedder.load()

    rungs: list[dict] = []
    for length in lengths:
        # 初始批量取自 token 预算，越长的序列自动压到越小的批；--batch 可强制指定
        default_batch = max(1, min(8, TOKEN_BUDGET // length))
        batch = args.batch or default_batch
        while batch >= 1:
            try:
                rung = run_rung(embedder, length, batch)
            except CUDA_OOM:
                torch.cuda.empty_cache()
                log.warning("length=%d batch=%d 触发 OOM，批大小减半重试", length, batch)
                batch //= 2
                continue
            rung["shrunk_from"] = default_batch
            rungs.append(rung)
            notes = []
            if batch != default_batch:
                notes.append("已降批")
            if rung["cpu_fallback"]:
                notes.append("已降级到 CPU，本档 GPU 不可行")
            print(
                f"  len {length:>5}  batch {batch:>2}  {rung['seconds']:>7.3f}s  "
                f"峰值 {rung['torch_peak_mib']:>5} MiB  设备增量 {rung['device_used_delta_mib']:>5} MiB"
                f"{'  (' + '；'.join(notes) + ')' if notes else ''}"
            )
            break

    recommendation = _recommend(rungs, settings.vram_limit_mib)
    payload = {
        "vram_total_mib": query_vram().total_mib,
        "vram_limit_mib": settings.vram_limit_mib,
        "rungs": rungs,
        "recommended": recommendation,
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n结果已写入 {args.out}")
    print("建议写回 settings.PRESETS 的档位（activation_mib 已含 20% 余量）：")
    for name, value in recommendation.items():
        print(f"  {name}: {value}")
    return 0


def _recommend(rungs: list[dict], vram_limit_mib: int) -> dict:
    """把实测峰值折算成建议档位：只保留峰值仍在硬上限内、且确实跑在 GPU 上的组合。

    跑在 CPU 上的一档说明 GPU 装不下，必须排除，否则会推荐一个不可行的档位。
    """
    recommendation: dict[str, dict] = {}
    for name, length in (("safe", 512), ("balanced", 1024), ("long", 8192)):
        feasible = [
            r
            for r in rungs
            if r["length"] == length
            and not r["cpu_fallback"]
            and r["torch_peak_mib"] <= vram_limit_mib
        ]
        if not feasible:
            recommendation[name] = {"max_length": length, "batch": 0, "note": "实测不通过"}
            continue
        best = max(feasible, key=lambda r: r["batch"])
        # 峰值含权重，档位的 activation_mib 只表达激活部分
        recommendation[name] = {
            "max_length": length,
            "batch": best["batch"],
            "activation_mib": int((best["torch_peak_mib"] - WEIGHTS_MIB) * HEADROOM),
        }
    return recommendation


if __name__ == "__main__":
    raise SystemExit(main())
