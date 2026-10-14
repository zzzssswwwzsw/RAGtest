"""preflight.py —— 加载模型前的显存/内存体检：量出可用量，不足则列出占用进程并引导清理。

硬约束：本模块只做探测与提示，绝不自动结束用户进程。
"""

from __future__ import annotations

import ctypes
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field, replace

from .settings import (
    CONTEXT_MIB,
    MIN_HOST_RAM_MIB,
    PRESETS,
    SAFETY_MARGIN_MIB,
    WEIGHTS_MIB,
    Preset,
    Settings,
    load_settings,
)

log = logging.getLogger("bge.preflight")

# WDDM 下图形进程常拒绝上报显存占用，用哨兵值区分"0 MiB"与"未上报"
UNKNOWN_MIB = -1

# NVML 用全 1 表示"该字段不可用"
_NVML_UNAVAILABLE = (2**64) - 1


# ── 数据结构 ──


@dataclass(frozen=True)
class ProcUsage:
    """单个进程的显存占用；used_mib 为 UNKNOWN_MIB 时表示驱动未上报。"""

    pid: int
    name: str
    used_mib: int


@dataclass(frozen=True)
class VramReport:
    """一次显存采样结果。"""

    total_mib: int
    used_mib: int
    free_mib: int
    procs: list[ProcUsage] = field(default_factory=list)
    source: str = "nvml"
    error: str = ""

    @property
    def known(self) -> bool:
        return self.total_mib > 0

    def to_dict(self) -> dict:
        return {
            "total_mib": self.total_mib,
            "used_mib": self.used_mib,
            "free_mib": self.free_mib,
            "source": self.source,
            "error": self.error,
            "processes": [
                {
                    "pid": p.pid,
                    "name": p.name,
                    "used_mib": None if p.used_mib == UNKNOWN_MIB else p.used_mib,
                }
                for p in self.procs
            ],
        }


@dataclass(frozen=True)
class Diagnosis:
    """体检结论，供 CLI 打印与 /preflight 接口复用。"""

    ok: bool
    need_mib: int
    free_mib: int
    vram: VramReport
    host_ram_ok: bool
    host_ram_avail_mib: int | None
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "need_mib": self.need_mib,
            "free_mib": self.free_mib,
            "reason": self.reason,
            "host_ram_ok": self.host_ram_ok,
            "host_ram_avail_mib": self.host_ram_avail_mib,
            "vram": self.vram.to_dict(),
        }


# ── 显存探测 ──


def query_vram() -> VramReport:
    """优先走 NVML，失败则退回 nvidia-smi，再失败返回零值报告。"""
    try:
        return _vram_via_nvml()
    except Exception as exc:  # noqa: BLE001 —— 探测失败不应阻断启动，转为降级路径
        log.debug("NVML 探测失败：%s", exc)
        nvml_error = str(exc)

    try:
        return _vram_via_smi()
    except Exception as exc:  # noqa: BLE001
        log.debug("nvidia-smi 探测失败：%s", exc)
        return VramReport(
            total_mib=0,
            used_mib=0,
            free_mib=0,
            source="unknown",
            error=f"NVML: {nvml_error}; nvidia-smi: {exc}",
        )


def _vram_via_nvml() -> VramReport:
    import pynvml  # noqa: PLC0415 —— 探测函数内延迟导入，缺库时走降级

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        procs = _nvml_processes(pynvml, handle)
        return VramReport(
            total_mib=int(mem.total // (1024 * 1024)),
            used_mib=int(mem.used // (1024 * 1024)),
            free_mib=int(mem.free // (1024 * 1024)),
            procs=procs,
            source="nvml",
        )
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001 —— 关闭失败不影响结果
            pass


def _nvml_processes(pynvml, handle) -> list[ProcUsage]:
    """合并计算进程与图形进程两张表，按显存降序返回。"""
    merged: dict[int, ProcUsage] = {}
    queries = (
        pynvml.nvmlDeviceGetComputeRunningProcesses,
        pynvml.nvmlDeviceGetGraphicsRunningProcesses,
    )
    for query in queries:
        try:
            entries = query(handle)
        except Exception as exc:  # noqa: BLE001 —— 单张表不可用不应中断体检
            log.debug("NVML 进程表 %s 不可用：%s", query.__name__, exc)
            continue
        for entry in entries:
            used = getattr(entry, "usedGpuMemory", _NVML_UNAVAILABLE)
            used_mib = UNKNOWN_MIB if used in (_NVML_UNAVAILABLE, None) else int(
                used // (1024 * 1024)
            )
            pid = int(entry.pid)
            prior = merged.get(pid)
            if prior is not None and prior.used_mib != UNKNOWN_MIB:
                continue  # 同一进程出现在两张表时保留有数据的那条
            merged[pid] = ProcUsage(pid=pid, name=_process_name(pid), used_mib=used_mib)

    return sorted(merged.values(), key=lambda p: (p.used_mib, p.pid), reverse=True)


def _vram_via_smi() -> VramReport:
    gpu = _run_smi(
        "--query-gpu=memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    )
    total, used, free = (int(x) for x in gpu.splitlines()[0].split(","))
    procs: list[ProcUsage] = []
    try:
        apps = _run_smi(
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        )
        for line in apps.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                procs.append(
                    ProcUsage(
                        pid=int(parts[0]),
                        name=os.path.basename(parts[1]),
                        used_mib=int(parts[2]),
                    )
                )
    except Exception as exc:  # noqa: BLE001 —— 进程表拿不到就只报总量
        log.debug("nvidia-smi 进程表不可用：%s", exc)

    return VramReport(
        total_mib=total,
        used_mib=used,
        free_mib=free,
        procs=sorted(procs, key=lambda p: p.used_mib, reverse=True),
        source="nvidia-smi",
    )


def _run_smi(*args: str) -> str:
    proc = subprocess.run(
        ["nvidia-smi", *args],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    return proc.stdout.strip()


def _process_name(pid: int) -> str:
    """取进程名：优先 NVML，其次 Win32 API；均失败则退回 PID 字样。"""
    try:
        import pynvml  # noqa: PLC0415

        name = pynvml.nvmlSystemGetProcessName(pid)
        if isinstance(name, bytes):  # 不同 pynvml 版本返回 str/bytes 不一致
            name = name.decode("utf-8", "replace")
        return os.path.basename(name)
    except Exception:  # noqa: BLE001 —— 新旧 NVML 差异较大，失败即降级
        pass

    if os.name == "nt":
        name = _win_process_name(pid)
        if name:
            return name
    return f"pid-{pid}"


def _win_process_name(pid: int) -> str:
    """Windows 兜底：OpenProcess + QueryFullProcessImageNameW，无第三方依赖。"""
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return ""
    try:
        size = ctypes.c_ulong(1024)
        buf = ctypes.create_unicode_buffer(size.value)
        if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return os.path.basename(buf.value)
    finally:
        kernel32.CloseHandle(handle)
    return ""


# ── 系统内存探测 ──


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def host_ram_avail_mib() -> int | None:
    """可用物理内存（MiB）；非 Windows 或调用失败返回 None（不阻断体检）。"""
    if os.name != "nt":
        return None
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.ullAvailPhys // (1024 * 1024))


# ── 判定与输出 ──


def diagnose(settings: Settings) -> Diagnosis:
    """比对"空闲显存"与"所需显存"，并附带系统内存检查。"""
    vram = query_vram()
    ram_avail = host_ram_avail_mib()
    ram_ok = ram_avail is None or ram_avail >= MIN_HOST_RAM_MIB

    if not vram.known:
        return Diagnosis(
            ok=False,
            need_mib=settings.need_mib,
            free_mib=0,
            vram=vram,
            host_ram_ok=ram_ok,
            host_ram_avail_mib=ram_avail,
            reason=f"无法读取显存信息（{vram.error}）",
        )

    free = vram.free_mib
    enough = free >= settings.need_mib
    reason = "" if enough else f"空闲 {free} MiB < 所需 {settings.need_mib} MiB"
    return Diagnosis(
        ok=enough,
        need_mib=settings.need_mib,
        free_mib=free,
        vram=vram,
        host_ram_ok=ram_ok,
        host_ram_avail_mib=ram_avail,
        reason=reason,
    )


def harden_console() -> None:
    """把输出编码与缓冲固定成可解释的状态，避免 Windows 下 GBK 编码炸掉整个体检流程。

    - 重定向到文件/管道：强制 UTF-8，日志才是项目规范的 UTF-8；并改为行缓冲，
      否则体检报告会在缓冲区里滞留，日志里顺序错乱，服务若硬崩则整段丢失；
    - 直接连终端：保留本机编码（Python 走控制台 API，中文照常显示）；
    - 两种情况都把不可编码字符降级成 '?'，而不是抛 UnicodeEncodeError。
    """
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


def _render_bar(used: int, total: int, width: int = 32) -> str:
    """条形图刻意用 ASCII：方块字符在 cp936 控制台上并不可靠。"""
    if total <= 0:
        return "?" * width
    filled = max(0, min(width, round(width * used / total)))
    return "#" * filled + "-" * (width - filled)


def format_report(settings: Settings, diag: Diagnosis, header: str) -> str:
    """把体检结论渲染成对齐的文本表格。"""
    vram = diag.vram
    lines = [
        header,
        f"  档位 {settings.preset.name}  序列上限 {settings.preset.max_length}"
        f"  批大小 {settings.preset.batch}  硬上限 {settings.vram_limit_mib} MiB",
        "",
    ]
    if vram.known:
        lines += [
            f"  显存总量 {vram.total_mib} MiB  [{_render_bar(vram.used_mib, vram.total_mib)}]"
            f"  已用 {vram.used_mib}  空闲 {vram.free_mib}  (来源 {vram.source})",
            f"  本次所需 {diag.need_mib} MiB = 权重 {WEIGHTS_MIB} + 上下文 {CONTEXT_MIB}"
            f" + 激活 {settings.preset.activation_mib} + 余量 {SAFETY_MARGIN_MIB}",
        ]
    else:
        lines.append(f"  显存信息不可用：{vram.error}")

    if diag.host_ram_avail_mib is not None:
        flag = "OK" if diag.host_ram_ok else "偏低"
        lines.append(
            f"  系统内存可用 {diag.host_ram_avail_mib} MiB（CPU 兜底门槛 {MIN_HOST_RAM_MIB}）[{flag}]"
        )

    if vram.procs:
        lines += ["", "  占用显存的进程（按占用降序）："]
        lines.append(f"    {'PID':>8}  {'进程':<28} {'显存':>10}")
        for proc in vram.procs[:15]:
            used = "未上报" if proc.used_mib == UNKNOWN_MIB else f"{proc.used_mib} MiB"
            lines.append(f"    {proc.pid:>8}  {_clip(proc.name, 28):<28} {used:>10}")
        if all(p.used_mib == UNKNOWN_MIB for p in vram.procs):
            lines.append("    提示：WDDM 驱动未上报单进程占用，需按进程名人工判断。")
    elif vram.known and vram.used_mib > 500:
        # 拿不到进程表时最容易误导用户：显存确实被占了，却说不出是谁
        lines.append(
            f"    未能列出占用进程（当前已用 {vram.used_mib} MiB），"
            "优先排查浏览器、视频客户端与截图/悬浮窗类工具。"
        )
    return "\n".join(lines)


def _clip(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


# ── 交互与降级 ──


def degrade(settings: Settings) -> Settings:
    """降到最保守档位：safe + batch=1，牺牲吞吐换稳定性。"""
    base: Preset = PRESETS["safe"]
    return replace(
        settings,
        preset=replace(base, name=f"{base.name}(forced)", batch=1),
    )


_PROMPT = """\
可选操作：
  [Enter] 我已清理，重新检测
  [s]     忽略警告，强制继续（自动降档到 safe 且 batch=1）
  [q]     退出
> """


def ensure_vram(
    settings: Settings,
    interactive: bool | None = None,
) -> Settings | None:
    """体检闸门：返回可用的配置；返回 None 表示用户选择退出。

    interactive 为 None 时按 stdin 是否为终端自动判断，服务模式下永不阻塞。
    """
    if interactive is None:
        interactive = not settings.force and sys.stdin.isatty()

    while True:
        diag = diagnose(settings)
        if diag.ok:
            print(format_report(settings, diag, "[ OK ] 显存体检通过，开始加载模型"))
            if not diag.host_ram_ok:
                log.warning(
                    "系统内存可用仅 %s MiB，低于 CPU 兜底门槛 %s MiB",
                    diag.host_ram_avail_mib,
                    MIN_HOST_RAM_MIB,
                )
            return settings

        print(format_report(settings, diag, "[FAIL] 显存不足，无法安全加载模型"))

        if not interactive:
            degraded = degrade(settings)
            log.warning(
                "非交互模式：跳过阻塞，自动降档到 %s 且 batch=%d（原因为 %s）",
                degraded.preset.name,
                degraded.preset.batch,
                diag.reason,
            )
            print(
                f"[WARN] 非交互模式：自动降档到 {degraded.preset.name}"
                f"（序列 {degraded.preset.max_length}，batch {degraded.preset.batch}）继续启动"
            )
            return degraded

        choice = input(_PROMPT).strip().lower()
        if choice == "q":
            print("已退出，未加载模型。")
            return None
        if choice == "s":
            degraded = degrade(settings)
            print(
                f"[WARN] 强制继续：已降档到 {degraded.preset.name}"
                f"（序列 {degraded.preset.max_length}，batch {degraded.preset.batch}）"
            )
            return degraded
        # 其余输入（含空回车）一律重新体检，形成循环


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：python -m src.preflight [--force] [--env-out PATH]。"""
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--help" in argv or "-h" in argv:
        print("用法：python -m src.preflight [--force] [--env-out PATH]")
        print("  --force        非交互模式，显存不足时自动降档继续")
        print("  --env-out PATH 把最终档位写成 env 文件（默认 .env.runtime）")
        return 0

    harden_console()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    force = "--force" in argv
    env_out = ".env.runtime"
    if "--env-out" in argv:
        env_out = argv[argv.index("--env-out") + 1]

    try:
        settings = load_settings(force=True if force else None)
    except ValueError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2

    result = ensure_vram(settings)
    if result is None:
        return 1

    # 落盘给 start.ps1：体检若降过档，uvicorn 必须继承降档后的档位，否则白降。
    # 档位名必须用 key（去掉展示后缀），否则写进去的 "safe(forced)" 不是合法档位，
    # 服务下次启动会直接因未知档位拒绝运行。
    lines = [
        f"BGE_PRESET={result.preset.key}",
        f"BGE_MAX_LENGTH={result.preset.max_length}",
        f"BGE_BATCH={result.preset.batch}",
        f"BGE_MEM_FRACTION={result.mem_fraction}",
    ]
    with open(env_out, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"已写出 {env_out}：{lines[0]} {lines[1]} {lines[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
