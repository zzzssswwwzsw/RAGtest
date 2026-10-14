"""check_api.py —— 从调用方视角验收服务：向量化契约 + 检索质量。

刻意不 import 仓库内任何模块，只通过 HTTP 打接口，因此可以单独拷给调用方使用。
服务需先启动（scripts/start.ps1），本脚本不负责拉起服务。

用法：
    python scripts/check_api.py
    python scripts/check_api.py --base http://127.0.0.1:8000 --timeout 60

退出码：0 全部通过；1 有失败项（可直接接进部署后的验收步骤）。
"""

from __future__ import annotations

import argparse
import base64
import sys
import time

import numpy as np
import requests

DIM = 1024
MAX_ITEMS = 256
# 批大小一变，FP16 的 reduction 顺序就变，末位会飘（实测同长同内容的三条批里
# 就有 2.4e-04 的分量偏差，与 padding、批内位置都无关）；所以跨批次只比方向，
# 不比逐位。方向偏差实测在 1e-5 量级，不影响任何排序。
ATOL_CROSS = 1e-3
ATOL_SAME = 1e-6
COS_MIN = 0.9999
NORM_TOL = 2e-3


# ── 输出 ──


def harden_console() -> None:
    """Windows 下管道/重定向默认按 ANSI 解码，中文报告会乱码，先固定成可解释状态。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            is_tty = stream.isatty()
        except (AttributeError, ValueError):
            is_tty = False
        if is_tty:
            reconfigure(errors="replace")
        else:
            reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


class Report:
    """收集断言结果，末尾统一给结论与退出码。"""

    def __init__(self) -> None:
        self.passed = 0
        self.failed: list[str] = []
        self.warned: list[str] = []

    def section(self, title: str) -> None:
        print(f"\n── {title} ──")

    def ok(self, name: str, detail: str = "") -> None:
        self.passed += 1
        print(f"  [PASS] {name}" + (f"  {detail}" if detail else ""))

    def fail(self, name: str, detail: str) -> None:
        self.failed.append(name)
        print(f"  [FAIL] {name}  {detail}")

    def check(self, name: str, passed: bool, detail: str = "") -> bool:
        self.ok(name, detail) if passed else self.fail(name, detail)
        return passed

    def warn(self, name: str, detail: str) -> None:
        self.warned.append(name)
        print(f"  [WARN] {name}  {detail}")


# ── 客户端 ──


class Client:
    def __init__(self, base: str, timeout: float) -> None:
        self.base = base.rstrip("/")
        self.timeout = timeout

    def post(self, payload: dict) -> requests.Response:
        return requests.post(f"{self.base}/v1/embeddings", json=payload, timeout=self.timeout)

    def get(self, path: str) -> requests.Response:
        return requests.get(f"{self.base}{path}", timeout=self.timeout)

    def embed(self, texts: list[str], **extra) -> np.ndarray:
        """发一批文本，返回 (n, DIM) float32；非 200 直接抛，由调用处统一兜。"""
        response = self.post({"input": texts, "model": "bge-m3", **extra})
        response.raise_for_status()
        return _vectors(response.json())


def _vectors(body: dict) -> np.ndarray:
    return np.asarray([item["embedding"] for item in body["data"]], dtype=np.float32)


# ── 断言 ──


def check_health(client: Client, rep: Report) -> dict | None:
    """确认服务活着，并取回档位上限供后面的截断断言使用。"""
    try:
        response = client.get("/healthz")
    except requests.exceptions.RequestException as exc:
        rep.fail("服务可达", f"连不上 {client.base}：{exc}")
        return None
    if response.status_code != 200:
        rep.fail("服务可达", f"/healthz 返回 HTTP {response.status_code}")
        return None

    health = response.json()
    rep.ok(
        "服务可达",
        f"preset={health.get('preset')} max_length={health.get('max_length')} "
        f"backend={health.get('backend')}",
    )
    degradations = health.get("degradations") or []
    if degradations:
        rep.warn("无降级记录", f"degradations 非空，已有请求走了 CPU 兜底：{degradations}")
    else:
        rep.ok("无降级记录", "degradations 为空")
    return health


def check_vectors(client: Client, rep: Report) -> None:
    """向量本身的形状、归一化、顺序对齐与确定性。"""
    texts = [
        "检索增强生成需要把文本编码成向量",
        "今天食堂的红烧肉有点咸",
        "向量数据库通常用余弦相似度做召回",
    ]
    response = client.post({"input": texts})
    body = response.json()
    V = _vectors(body)

    rep.check("输出形状 (n, 1024)", V.shape == (len(texts), DIM), f"实测 {V.shape}")

    index = [item["index"] for item in body["data"]]
    rep.check(
        "index 与输入顺序一一对应",
        index == list(range(len(texts))),
        f"index={index}",
    )

    norms = np.linalg.norm(V, axis=1)
    rep.check(
        "向量已 L2 归一化",
        bool(np.all(np.abs(norms - 1.0) < NORM_TOL)),
        f"范数 {norms.min():.6f} ~ {norms.max():.6f}",
    )

    rep.check(
        "不同文本得到不同向量",
        not np.allclose(V[0], V[1]),
        f"两条相似度 {float(V[0] @ V[1]):+.4f}",
    )

    # 同一文本单独发 vs 混在批里发，方向必须一致（比逐位宽松，理由见常量注释）
    single = client.embed([texts[0]])[0]
    cosine = float(single @ V[0])
    rep.check(
        "单条与批量的方向一致",
        cosine > COS_MIN,
        f"余弦 {cosine:.6f}，分量偏差 {float(np.abs(single - V[0]).max()):.2e}（FP16 批效应，正常）",
    )

    # 打乱输入顺序，"1 条 vs 2 条"这类错位最容易在这里暴露
    shuffled = [texts[2], texts[0], texts[1]]
    Vs = client.embed(shuffled)
    rep.check(
        "乱序输入能按 index 正确还原",
        bool(np.allclose(Vs[0], V[2], atol=ATOL_CROSS) and np.allclose(Vs[1], V[0], atol=ATOL_CROSS)),
        f"偏差 {float(np.abs(Vs[0] - V[2]).max()):.2e} / {float(np.abs(Vs[1] - V[0]).max()):.2e}",
    )

    # 同形同批重复请求：不做缓存也不应该飘。
    # 注意"可复现"的前提是批量组成相同，批大小一换就不是逐位复现了（见常量注释）。
    again = client.embed([texts[0]])[0]
    rep.check(
        "同一输入、同批量组成下逐位可复现",
        bool(np.allclose(again, single, atol=ATOL_SAME)),
        f"两次偏差 {float(np.abs(again - single).max()):.2e}",
    )


def check_encoding(client: Client, rep: Report) -> None:
    """base64 编码路径必须与 float 逐位等价，否则落库会悄悄取到错数据。"""
    sample = "base64 往返一致性验证"
    as_float = client.post({"input": sample})
    as_base64 = client.post({"input": sample, "encoding_format": "base64"})

    vf = np.asarray(as_float.json()["data"][0]["embedding"], dtype=np.float32)
    payload = as_base64.json()["data"][0]["embedding"]
    vb = np.frombuffer(base64.b64decode(payload), dtype=np.float32)

    rep.check(
        "base64 与 float 逐位一致",
        bool(np.array_equal(vf, vb)),
        f"dtype={vb.dtype} shape={vb.shape}",
    )
    rep.check(
        "base64 体积更小",
        len(as_base64.content) < len(as_float.content),
        f"float {len(as_float.content)} B -> base64 {len(as_base64.content)} B",
    )


def check_limits(client: Client, rep: Report) -> None:
    """档位上限与条数上限：这两条最容易在调用方那边被当成"偶发失败"。"""
    accepted = client.post({"input": ["x"] * MAX_ITEMS})
    rep.check(
        f"{MAX_ITEMS} 条可接受",
        accepted.status_code == 200,
        f"HTTP {accepted.status_code}",
    )

    rejected = client.post({"input": ["x"] * (MAX_ITEMS + 1)})
    detail = _detail(rejected)
    rep.check(
        f"{MAX_ITEMS + 1} 条被拒 (400)",
        rejected.status_code == 400,
        f"HTTP {rejected.status_code} {detail}",
    )


def check_truncation(client: Client, rep: Report, max_length: int | None) -> None:
    """超长文本是静默截断的，必须能从 usage 上看出来。"""
    if not max_length:
        rep.warn("超长文本截断", "档位未上报 max_length，跳过")
        return

    long_text = "长文本截断验证。" * (max_length * 4)
    body = client.post({"input": long_text}).json()
    tokens = body["usage"]["prompt_tokens"]
    rep.check(
        "超长文本截断到档位上限",
        tokens == max_length,
        f"{len(long_text)} 字符 -> prompt_tokens={tokens}（档位上限 {max_length}）",
    )


def check_empty(client: Client, rep: Report) -> None:
    """空串不报错、返回一个毫无语义的归一化向量——已知行为，必须提醒调用方。"""
    body = client.post({"input": ["正常文本", ""]}).json()
    V = _vectors(body)
    rep.warn(
        "空串返回可用向量（已知行为）",
        f"空串 L2={float(np.linalg.norm(V[1])):.4f}，与正常文本相似度 "
        f"{float(V[0] @ V[1]):+.4f} —— 调用方须自行过滤空串与纯空白",
    )


# 每条用例标注期望命中的文档下标，用来验证检索排序是否真的对
RETRIEVAL_CASES: list[dict] = [
    {
        "query": "怎么把文本变成向量做召回",
        "docs": [
            "向量数据库通常用余弦相似度或内积做召回，HNSW 是最常见的索引结构。",
            "今天食堂的红烧肉有点咸，下次少放点酱油。",
            "Windows 下 PowerShell 读取无 BOM 的 UTF-8 文件会按 ANSI 解码，中文会乱码。",
            "bge-m3 支持稠密、稀疏与多向量三种检索方式。",
        ],
        "expect": 0,
    },
    {
        "query": "启动时显存不够怎么办",
        "docs": [
            "今天食堂的红烧肉有点咸。",
            "加载模型前会做显存体检，不足时列出占用进程并引导清理，绝不自动结束他人进程。",
            "bge-m3 输出 1024 维稠密向量。",
        ],
        "expect": 1,
    },
    {
        "query": "向量是多少维的",
        "docs": [
            "PowerShell 5.1 会把无 BOM 的 UTF-8 当 ANSI 解码。",
            "bge-m3 输出 1024 维稠密向量，已做 L2 归一化。",
            "红烧肉有点咸。",
        ],
        "expect": 1,
    },
]


def check_retrieval(client: Client, rep: Report) -> None:
    """端到端：文档与 query 同批编码后按内积排序，看标注的正确答案是否排第一。"""
    for case in RETRIEVAL_CASES:
        body = client.post({"input": case["docs"] + [case["query"]]}).json()
        V = _vectors(body)
        scores = V[:-1] @ V[-1]  # 已归一化，内积即余弦
        order = np.argsort(-scores)
        top = int(order[0])
        ranking = "  ".join(f"doc{i}:{scores[i]:+.4f}" for i in order)
        if top == case["expect"]:
            rep.ok(f"检索命中「{case['query']}」", f"Top1=doc{top}  {ranking}")
        else:
            rep.fail(
                f"检索命中「{case['query']}」",
                f"期望 doc{case['expect']}，实得 doc{top}  {ranking}",
            )


# ── 入口 ──


def _detail(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:120]
    detail = body.get("detail", body)
    return str(detail)[:120]


def main() -> int:
    parser = argparse.ArgumentParser(description="调用方视角验收 bge-m3 嵌入服务")
    parser.add_argument("--base", default="http://127.0.0.1:8000", help="服务地址")
    parser.add_argument("--timeout", type=float, default=120.0, help="单请求超时秒数")
    args = parser.parse_args()

    harden_console()
    client = Client(args.base, args.timeout)
    rep = Report()
    started = time.perf_counter()

    print(f"验收目标：{client.base}（超时 {args.timeout:.0f}s）")

    rep.section("1/4 服务状态")
    health = check_health(client, rep)
    if health is None:
        return _finish(rep, started)

    try:
        rep.section("2/4 向量化契约")
        check_vectors(client, rep)

        rep.section("3/4 编码与边界")
        check_encoding(client, rep)
        check_truncation(client, rep, health.get("max_length"))
        check_limits(client, rep)
        check_empty(client, rep)

        rep.section("4/4 检索质量")
        check_retrieval(client, rep)
    except requests.exceptions.RequestException as exc:
        rep.fail("请求异常中断", str(exc))

    return _finish(rep, started)


def _finish(rep: Report, started: float) -> int:
    elapsed = time.perf_counter() - started
    print(f"\n{'=' * 56}")
    if rep.failed:
        print(f"结论：失败 {len(rep.failed)} 项 / 通过 {rep.passed} 项，耗时 {elapsed:.2f}s")
        for name in rep.failed:
            print(f"  - {name}")
        return 1
    print(f"结论：全部通过（{rep.passed} 项）", end="")
    print(f"，警告 {len(rep.warned)} 项，耗时 {elapsed:.2f}s" if rep.warned else f"，耗时 {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
