"""pipeline.py —— 数据集测试：原文本 -> 切块 -> 向量化 -> 查询匹配度。

三个数据目录（缺失时脚本自动创建）：

    data/raw/         原文本，每篇一个 .txt。人工维护，入库。
    data/chunks/      切割后的文本，每篇一个 .txt，块之间以
                      "===== chunk 0001 =====" 分隔。派生文件，gitignore。
    data/embeddings/  向量，每篇一个 .npy，float32 形状 (块数, 1024)。派生文件，gitignore。

--stage 决定从哪一步开始，也就是用哪份数据：

    raw         切块 -> 向量化 -> 查询
    chunks      向量化 -> 查询
    embeddings  跳过编码，直接查询

用法（服务需先启动）：

    python scripts/pipeline.py --stage raw --query "你的问题"
    python scripts/pipeline.py --stage raw                 # 不带 --query 则进交互模式
    python scripts/pipeline.py --stage embeddings --top-k 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CHUNK_DIR = DATA_DIR / "chunks"
VECTOR_DIR = DATA_DIR / "embeddings"

# 服务端单请求上限，见 src/api.py 的 MAX_INPUT_ITEMS
MAX_ITEMS = 256
DIM = 1024

# ══════════════════════════════════════════════════════════════════════
#  切割参数：要改「块大小 / 重叠 / 切割规则」，只需要改这一段
# ══════════════════════════════════════════════════════════════════════

# 单块最大字符数。中文大约 1 字 1 token，直接按字符数近似 token 数。
# 不得超过当前档位的 max_length（默认档位 512），否则服务端会静默截断，
# 块尾的内容等于白切 —— 启动时脚本会拿 /healthz 的 max_length 校验这一点。
CHUNK_SIZE = 400

# 相邻块之间保留的重叠字符数。切点落在句子中间时，靠它保住跨块的语义。
CHUNK_OVERLAP = 50

# 优先级从强到弱的分隔符：先按空行分段，再按换行，再按句末标点，最后按空格。
# 想改成「整篇不切」就把 CHUNK_SIZE 调到比最长文本还大；想改切分习惯就动这个元组。
CHUNK_SEPARATORS = ("\n\n", "\n", "。", "！", "？", "；", ". ", "! ", "? ", "; ", " ")

# 块分隔标记，同时也是 chunks/*.txt 的解析依据
CHUNK_MARK = "===== chunk "

# ══════════════════════════════════════════════════════════════════════


# ── 输出 ──


def _is_tty(stream) -> bool:
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


def harden_console() -> None:
    """Windows 下管道/重定向默认按 ANSI 解码，中文会乱码，先固定成可解释状态。

    终端保持控制台编码（中文本来就能正常显示），只管错误处理；
    管道/重定向一律按 UTF-8，日志与查询文本才是项目规范里的那个编码。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        if _is_tty(stream):
            reconfigure(errors="replace")
        else:
            reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

    # stdin 也要管：非终端时 Python 按区域编码解输入，从管道喂进来的 UTF-8
    # 查询会被解成乱码，算出来的相似度全是噪声，而且不会报任何错。
    reconfigure = getattr(sys.stdin, "reconfigure", None)
    if reconfigure is not None and not _is_tty(sys.stdin):
        reconfigure(encoding="utf-8", errors="replace")


def info(message: str) -> None:
    print(f"  {message}")


def step(title: str) -> None:
    print(f"\n── {title} ──")


# ── 切块 ──


def _split_recursive(text: str, separators: tuple[str, ...], size: int) -> list[str]:
    """按分隔符优先级递归拆分，拆不动了才硬切，保证每段不超过 size。"""
    if len(text) <= size:
        return [text]
    if not separators:
        return [text[i : i + size] for i in range(0, len(text), size)]

    separator, rest = separators[0], separators[1:]
    pieces = []
    # 把分隔符留在前一段末尾，否则句末标点会被吃掉
    parts = text.split(separator)
    for index, part in enumerate(parts):
        piece = part + separator if index < len(parts) - 1 else part
        if not piece:
            continue
        pieces.extend(
            [piece] if len(piece) <= size else _split_recursive(piece, rest, size)
        )
    return pieces


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """把一篇文本切成块：先按结构切，再按 size 合并，相邻块保留 overlap 个字符。

    被丢弃的只有纯空白块 —— 空串送进服务端会得到一个"范数为 1 但毫无语义"的向量，
    混进库里会污染召回，所以在切块这一步就滤掉。
    """
    pieces = [p for p in _split_recursive(text, CHUNK_SEPARATORS, size) if p.strip()]

    chunks: list[str] = []
    buffer = ""
    for piece in pieces:
        if buffer and len(buffer) + len(piece) > size:
            chunks.append(buffer)
            buffer = buffer[-overlap:] if overlap > 0 else ""
        buffer += piece
    if buffer.strip():
        chunks.append(buffer)
    return chunks


def write_chunks(path: Path, chunks: list[str]) -> None:
    lines: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        lines.append(f"{CHUNK_MARK}{index:04d} =====")
        lines.append(chunk)
    # 用 open 而不是 Path.write_text：后者的 newline 参数要 Python 3.10+，
    # 而这个脚本是给调用方用的，不该绑死在某个小版本上（本机 PATH 上就是 3.8）。
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def read_chunks(path: Path) -> list[str]:
    chunks: list[str] = []
    current: list[str] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(CHUNK_MARK):
            if current is not None:
                chunks.append("\n".join(current))
            current = []
        elif current is not None:
            current.append(line)
    if current is not None:
        chunks.append("\n".join(current))
    return [c for c in chunks if c.strip()]


# ── 服务调用 ──


class Client:
    def __init__(self, base: str, timeout: float) -> None:
        self.base = base.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict:
        response = requests.get(f"{self.base}/healthz", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def embed(self, texts: list[str]) -> np.ndarray:
        """按服务端上限分片编码，返回 (n, 1024) float32。"""
        out: list[list[float]] = []
        for start in range(0, len(texts), MAX_ITEMS):
            batch = texts[start : start + MAX_ITEMS]
            response = requests.post(
                f"{self.base}/v1/embeddings",
                json={"input": batch, "model": "bge-m3"},
                timeout=self.timeout,
            )
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}: {_detail(response)}")
            out.extend(item["embedding"] for item in response.json()["data"])
        return np.asarray(out, dtype=np.float32)


def _detail(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:120]
    return str(body.get("detail", body))[:120]


# ── 三个阶段 ──


def stage_chunk(raw_files: list[Path]) -> list[Path]:
    """raw/*.txt -> chunks/*.txt，返回写出的块文件列表。"""
    written: list[Path] = []
    for raw in raw_files:
        chunks = split_text(raw.read_text(encoding="utf-8"))
        if not chunks:
            info(f"{raw.name}: 空文件，跳过")
            continue
        target = CHUNK_DIR / f"{raw.stem}.txt"
        write_chunks(target, chunks)
        info(f"raw/{raw.name} -> {CHUNK_DIR.name}/{target.name}  切出 {len(chunks)} 块")
        written.append(target)
    return written


def prune_orphans(keep: set[str]) -> None:
    """清掉 raw/ 里已经不存在的块文件与向量文件。

    只按 raw/ 新增是危险的：删掉一篇原文本后，它的旧块仍在 chunks/ 里，
    会被照常读进语料，检索结果里就混着已经删掉的文档。raw/ 是权威来源。
    """
    orphans = [
        path
        for path in sorted(CHUNK_DIR.glob("*.txt")) + sorted(VECTOR_DIR.glob("*.npy"))
        if path.stem not in keep
    ]
    for path in orphans:
        path.unlink()
        info(f"清理孤儿文件 {path.parent.name}/{path.name}")


def stage_embed(client: Client, chunk_files: list[Path], force: bool) -> None:
    """chunks/*.txt -> embeddings/*.npy。已是最新的默认跳过，--force 强制重算。"""
    for chunk_file in chunk_files:
        target = VECTOR_DIR / f"{chunk_file.stem}.npy"
        if not force and target.exists() and target.stat().st_mtime > chunk_file.stat().st_mtime:
            info(f"{VECTOR_DIR.name}/{target.name}: 已是最新，跳过（--force 可强制重算）")
            continue

        chunks = read_chunks(chunk_file)
        if not chunks:
            info(f"{chunk_file.name}: 没有有效块，跳过")
            continue

        started = time.perf_counter()
        vectors = client.embed(chunks)
        np.save(target, vectors)
        info(
            f"{CHUNK_DIR.name}/{chunk_file.name} -> {VECTOR_DIR.name}/{target.name}  "
            f"{vectors.shape[0]} 块 {vectors.shape[1]} 维  耗时 {time.perf_counter() - started:.2f}s"
        )


def load_corpus() -> tuple[list[str], list[tuple[str, int]]]:
    """读回全部块文本与出处，供查询使用。返回 (文本列表, [(文件名, 块序号), ...])。"""
    texts: list[str] = []
    origins: list[tuple[str, int]] = []
    for chunk_file in sorted(CHUNK_DIR.glob("*.txt")):
        vector_path = VECTOR_DIR / f"{chunk_file.stem}.npy"
        if not vector_path.exists():
            info(f"跳过 {chunk_file.name}：缺少同名向量文件，先跑 --stage chunks")
            continue
        chunks = read_chunks(chunk_file)
        vectors = np.load(vector_path)
        if len(chunks) != vectors.shape[0]:
            raise RuntimeError(
                f"{chunk_file.name} 有 {len(chunks)} 块，但 {vector_path.name} 有 "
                f"{vectors.shape[0]} 条向量 —— 文本改过而向量没重算，"
                f"请重跑 --stage chunks（或加 --force）"
            )
        texts.extend(chunks)
        origins.extend((chunk_file.stem, index) for index in range(1, len(chunks) + 1))
    return texts, origins


def load_vectors() -> np.ndarray:
    matrices = [np.load(path) for path in sorted(VECTOR_DIR.glob("*.npy"))]
    if not matrices:
        return np.zeros((0, DIM), dtype=np.float32)
    return np.vstack(matrices)


def answer(client: Client, query: str, top_k: int) -> None:
    texts, origins = load_corpus()
    matrix = load_vectors()
    if not texts or matrix.shape[0] != len(texts):
        print("  没有可查的数据，先跑 --stage raw 或 --stage chunks")
        return

    vector = client.embed([query])[0]
    scores = matrix @ vector  # 向量已 L2 归一化，内积即余弦
    order = np.argsort(-scores)[:top_k]

    print(f"  查询：{query}")
    for rank, index in enumerate(order, start=1):
        name, number = origins[index]
        preview = " ".join(texts[index].split())[:46]
        print(f"    {rank}. {scores[index]:+.4f}  [{name} #{number}]  {preview}")


# ── 入口 ──


def main() -> int:
    parser = argparse.ArgumentParser(
        description="数据集测试：原文本 -> 切块 -> 向量化 -> 查询匹配度",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=("raw", "chunks", "embeddings"),
        default="raw",
        help="从哪一步开始：raw=切块+向量化，chunks=只向量化，embeddings=直接查询",
    )
    parser.add_argument("--base", default="http://127.0.0.1:8000", help="服务地址")
    parser.add_argument("--timeout", type=float, default=120.0, help="单请求超时秒数")
    parser.add_argument("--query", default="", help="查询文本；省略则进交互模式")
    parser.add_argument("--top-k", type=int, default=5, help="返回最相似的块数")
    parser.add_argument("--force", action="store_true", help="忽略时间戳，强制重新向量化")
    args = parser.parse_args()

    harden_console()
    for directory in (RAW_DIR, CHUNK_DIR, VECTOR_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    client = Client(args.base, args.timeout)

    print(f"数据集目录：{DATA_DIR}")
    try:
        health = client.health()
    except requests.exceptions.RequestException as exc:
        print(f"  连不上服务 {client.base}：{exc}")
        print("  先启动服务：powershell -File scripts/start.ps1")
        return 1
    info(f"服务已就绪  preset={health.get('preset')} max_length={health.get('max_length')}")

    # 块比档位上限还长就会被静默截断，切了等于白切
    max_length = health.get("max_length")
    if max_length and CHUNK_SIZE > max_length:
        print(
            f"  警告：CHUNK_SIZE={CHUNK_SIZE} 超过档位上限 max_length={max_length}，"
            f"超出部分会被服务端静默截断，建议调小"
        )

    if args.stage == "raw":
        step("1/3 切块")
        raw_files = sorted(RAW_DIR.glob("*.txt"))
        if not raw_files:
            print(f"  {RAW_DIR} 里没有 .txt，放几篇原文本再跑")
            return 1
        info(f"共 {len(raw_files)} 篇：{', '.join(f.name for f in raw_files)}")
        prune_orphans({raw.stem for raw in raw_files})
        stage_chunk(raw_files)

        step("2/3 向量化")
        stage_embed(client, sorted(CHUNK_DIR.glob("*.txt")), args.force)
    elif args.stage == "chunks":
        step("1/2 向量化")
        chunk_files = sorted(CHUNK_DIR.glob("*.txt"))
        if not chunk_files:
            print(f"  {CHUNK_DIR} 里没有块文件，先跑 --stage raw")
            return 1
        stage_embed(client, chunk_files, args.force)

    texts, _ = load_corpus()
    matrix = load_vectors()
    # 步数随 stage 变：raw 走三步，chunks 走两步，embeddings 只有查询
    step({"raw": "3/3 查询", "chunks": "2/2 查询", "embeddings": "查询"}[args.stage])
    info(f"语料 {len(texts)} 块，向量 {matrix.shape}")

    if args.query:
        answer(client, args.query, args.top_k)
        return 0

    # 交互模式：反复输入查询，:q 退出
    print("  输入查询文本回车查看匹配度，输入 :q 退出\n")
    while True:
        try:
            query = input("查询> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query in (":q", ":quit", "q", "exit"):
            break
        if not query:
            continue
        answer(client, query, args.top_k)
    return 0


if __name__ == "__main__":
    # 语料与向量对不上、服务返回非 200 这类问题抛的是 RuntimeError，
    # 裸 traceback 对使用者没用，折成一行结论就够了。
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"\n错误：{exc}", file=sys.stderr)
        sys.exit(1)
