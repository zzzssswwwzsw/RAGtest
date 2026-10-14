# BGE-M3 CUDA 本地部署执行方案（RTX 3060 Laptop 6GB）

## 一、Summary

在本机（RTX 3060 Laptop / 6GB 显存 / 16GB 内存 / Windows）用 CUDA 部署 `BAAI/bge-m3` 稠密向量 embedding，
以**常驻 FastAPI 服务**形态对外提供 OpenAI 兼容的 `/v1/embeddings` 接口，仅输出 1024 维 dense 向量。
核心目标是**留足安全冗余、绝不 OOM 崩掉系统**，并在**加载模型前做显存体检 + 交互式提示清理占用**。

## 二、Current State Analysis（实测结论）

| 项目 | 实测值 | 说明 |
|---|---|---|
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU | WDDM 模式 |
| **显存总量** | **6144 MiB（6GB）** | ⚠️ 与用户认知的 "8G" 不符，3060 Laptop 只有 6GB 版，显存预算按 6GB 算 |
| 已用显存（当前） | 1259 MiB | Edge / 抖音(D:\DY\douyin) / Copilot / Trae / NVIDIA Overlay 等十几个进程 |
| 空闲可用 | ≈ 4885 MiB | 清理占用后可提升到 ≈ 5300–5800 MiB |
| 驱动 / CUDA | 552.22 / CUDA 12.4 | 可跑 cu121 运行时（向下兼容） |
| 系统内存 | 16GB | CPU 兜底时可用 |
| 磁盘 | C: 220GB 空闲 / D: 509GB 空闲 | conda 环境与模型权重放 D 盘 |
| Python | 3.8.10 (base) / conda 4.10.3 / pip 21.1.3 | 版本偏旧，故**新建独立环境** |
| torch | 未安装 | 全新安装 |

工作目录 `d:\1_AAA_My_Project\RAGtest` 当前为**空目录**，无历史代码、无 git 仓库。

## 三、显存预算与安全冗余设计（方案的量化依据）

bge-m3 = XLM-RoBERTa-large（约 568M 参数），仅取 dense 输出（不启用 sparse / ColBERT 多向量）：

| 组成 | FP32 | **FP16（本方案采用）** |
|---|---|---|
| 模型权重 | ≈ 2.3 GB | **≈ 1.15 GB** |
| CUDA 上下文 + cuBLAS/cuDNN 工作区 | ≈ 0.5 GB | ≈ 0.5 GB |
| 前向激活（bs=8, len=512, inference_mode） | ≈ 1.5 GB | ≈ 0.6 GB |
| **合计峰值** | ≈ 4.3 GB（贴边，危险） | **≈ 2.3 GB（安全）** |

**三重冗余机制（必须全部落地）**

1. **FP16 推理**：显存直接减半，是 6GB 卡能跑稳的前提。绝对不加载 FP32。
2. **进程级硬上限**：`torch.cuda.set_per_process_memory_fraction(0.62)` → 硬顶 **≈ 3810 MiB**。
   超出时由 PyTorch 抛 `OutOfMemoryError`（可捕获、可降级），而**不是**触发 WDDM 向共享内存换页拖死整机。
   此时留 **≈ 2.3 GB** 给桌面程序，满足"硬件允许范围内留安全冗余"的要求。
3. **token 预算动态批**：按 `batch × length ≤ 8192 tokens` 组批（如 16×512 或 1×8192）。
   长文本时自动把 batch 压到 1，规避注意力矩阵 O(n²) 的显存爆炸。
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 抑制碎片化。

**档位预设**（`settings.py` 中固化，可用环境变量覆盖）

| 档位 | max_length | batch | 适用 |
|---|---|---|---|
| `safe`（默认） | 512 | 8 | RAG 常规分块，最稳 |
| `balanced` | 1024 | 4 | 中等长度段落 |
| `long` | 8192 | 1 | 超长文本，全长度支持 |

## 四、Skill 选型（按用户规则 3）

**结论：本任务不使用任何内置或插件 Skill。**
理由：纯后端 Python 工程任务（环境搭建 + 模型推理 + HTTP 服务），与现有 Skill 池（浏览器自动化、
代码审查、文档/PPT/表格生成、飞书、小程序等）均无匹配项。仅使用 PowerShell 与 Python 工具链。
后续若需要压测报告可视化，再考虑 `html-report`。

## 五、Proposed Changes（逐文件）

目录结构（全部新建于 `d:\1_AAA_My_Project\RAGtest`）：

```
.trae/documents/bge-m3-cuda-embedding-plan.md   本方案（已存在）
.gitignore                                       忽略 models/、.venv、__pycache__ 等大文件与缓存
requirements.txt                                 锁定依赖版本
src/__init__.py
src/settings.py                                  配置与三档预设，读取环境变量覆盖
src/preflight.py                                 ⭐ 显存/内存体检 + 交互式清理提示（本次新增重点）
src/embedder.py                                  模型封装：FP16 加载 / dense 编码 / OOM 自动降级
src/api.py                                       FastAPI 服务：/v1/embeddings、/healthz、/preflight
scripts/start.ps1                                体检 → 启动服务（设置全部环境变量）
scripts/bench.py                                 压测，实测本机可承受的最大 batch × length
tests/test_smoke.py                              冒烟测试
```

### 1. `requirements.txt`

```
torch==2.3.1+cu121        # 适配驱动 552.22 / CUDA 12.4（向下兼容 12.1 运行时）
FlagEmbedding==1.2.11     # 官方 bge-m3 封装，use_fp16=True
transformers==4.40.2
numpy==1.26.4             # 必须 <2.0，FlagEmbedding 与 numpy2 不兼容
fastapi==0.111.0
uvicorn==0.30.1
pydantic==2.7.1
nvidia-ml-py==12.535.133  # pynvml，nvidia-smi 的库封装，用于体检
```
结构：**what** 锁定版本 / **why** Python 3.10 全新环境，避免 3.8 的兼容坑与 base 环境污染。

### 2. `src/settings.py`

- 固化三档预设（上表 safe/balanced/long）。
- 常量：`VRAM_TOTAL_MIB=6144`、`MEM_FRACTION=0.62`、`TOKEN_BUDGET=8192`、`SAFETY_MARGIN_MIB=800`。
- 全部支持环境变量覆盖（`BGE_PRESET`、`BGE_MAX_LENGTH`、`BGE_BATCH`、`BGE_MEM_FRACTION`）。

### 3. `src/preflight.py` ⭐（对应你新增的需求）

功能：**在加载模型前检测显存，不足则列出占用进程并提示清理，循环直到满足。**

- 用 `pynvml` 读取：总量 / 已用 / 空闲 / 各进程占用（进程名 + PID + MiB）。
- 需要量估算 = 权重(1150) + CUDA上下文(500) + 激活预算(预设档位对应值) + `SAFETY_MARGIN_MIB(800)`。
- 判定：
  - `空闲 ≥ 需要量` → 打印 ✅ 表格，直接进入加载。
  - `空闲 < 需要量` → 打印 ❌ 诊断表格（**按占用显存降序列出 Top 进程并给出建议关闭项**），并提示：
    ```
    [Enter] 我已清理，重新检测
    [s]     忽略警告，强制继续（自动降档到 safe 且 batch=1）
    [q]     退出
    ```
    输入 Enter 则重新检测，形成循环。
- 服务模式（`start.ps1` 非交互 / `--force` 场景）下不阻塞 stdin：仅打印警告日志，并自动降档到 `safe` + `batch=1`，同时给出明确提示。
- 同一函数复用为 `GET /preflight` 接口，运行中也可随时查询冗余情况。
- 附带系统内存检查：若可用内存 < 4GB 则提示（CPU 兜底需要约 3.5GB RAM）。

### 4. `src/embedder.py`

- `BGEM3FlagModel(model_path, use_fp16=True)` 加载；**只取 dense**：
  `encode(..., return_dense=True, return_sparse=False, return_colbert_vecs=False)`。
- `torch.inference_mode()` 包裹前向，`torch.cuda.empty_cache()` 在空闲时调用。
- **OOM 自愈**：捕获 `torch.cuda.OutOfMemoryError` → `empty_cache()` → batch 减半重试 → 仍失败则
  该请求转 CPU（fp32）处理并标记告警，服务本身**永不崩溃**。
- 按 `TOKEN_BUDGET` 做动态组批（内部排序后按 token 总量装箱）。
- 全局信号量限制并发为 2，避免多请求叠加撑爆显存。

### 5. `src/api.py`

- `uvicorn` **单 worker**（多 worker 会在显存中复制多份模型，必须禁止）。
- 接口：
  - `POST /v1/embeddings`：OpenAI 兼容（入参 `input`(str/list)、`model`、可选 `max_length`），
    返回 `data[].embedding`（1024 维）+ `usage`。
  - `GET /healthz`：存活 + 当前档位 + 显存占用/上限。
  - `GET /preflight`：实时显存体检 JSON。
- 入参超限时返回 400；显存不足且无法降级时返回 `507 Insufficient Storage`（而非 500）。

### 6. `scripts/start.ps1`

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:CUDA_VISIBLE_DEVICES      = "0"
$env:HF_ENDPOINT               = "https://hf-mirror.com"   # 国内网络加速
$env:HF_HOME                   = "D:\1_AAA_My_Project\RAGtest\models"  # 权重落 D 盘
python -m src.preflight        # ⭐ 先体检、再启动
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --workers 1
```

### 7. `scripts/bench.py`

- 按 token 预算递增（512/1024/2048/4096/8192）逐档实测峰值显存与耗时，输出表格，
  **用实测数据确认/微调上表的 batch 档位**，而非拍脑袋。含单次 OOM 自动收缩逻辑。

### 8. 环境与仓库

- 新环境放 D 盘避免吃 C 盘：`conda create -p D:\envs\bgem3 python=3.10 -y`，再 `pip install -r requirements.txt`。
- 按用户规则 1、2：`git init` + `.gitignore`（忽略 `models/`、`*.pt`、`__pycache__/`、`.venv/`），
  里程碑提交采用 Conventional Commits，例如 `feat: 初始化 bge-m3 CUDA embedding 服务`。

## 六、Assumptions & Decisions

| 决策 | 理由 |
|---|---|
| 显存按 **6GB** 而非 8GB 规划 | nvidia-smi 实测 6144 MiB |
| 服务常驻 + 单 worker | 用户选择"常驻 API 服务"；多 worker 会翻倍显存 |
| 仅 dense 输出 | 用户选择；省显存，且 ColBERT 多向量在 6GB 长文本下风险高 |
| 新建 Python 3.10 conda 环境（前缀在 D 盘） | 用户选择；规避 3.8 依赖兼容问题 |
| FP16 + 0.62 硬上限 + token 预算批 三重冗余 | 满足"留有安全冗余、避免 OOM" |
| 不做自动杀进程，只提示 | 尊重用户手动清理的选择，避免误杀 |
| 优先 `BGEM3FlagModel` | 官方推荐路径，pooling（CLS）+ L2 归一化正确；`transformers` 直调作为备选 |
| 模型权重与 conda 环境均落 D 盘 | C 盘仅 220GB 空闲，D 盘 509GB |

## 七、Verification（验收步骤）

1. **环境**：`python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
   → 期望 `2.3.1+cu121 True NVIDIA GeForce RTX 3060 Laptop GPU`。
2. **体检**：`python -m src.preflight` → 输出显存表格；人为打开 Edge/抖音复现"不足"分支，
   验证提示与循环重测、`s`/`q` 分支均生效。
3. **压测定档**：跑 `scripts/bench.py`，确认各档峰值显存均 < 3810 MiB，且 8192 长度可跑通。
4. **加载峰值**：加载模型后 `nvidia-smi` 记录峰值，确认稳态占用 ≤ 3.0 GB、冗余 ≥ 1.5 GB。
5. **接口**：启动服务后
   `curl POST /v1/embeddings -d '{"input":"测试文本","model":"bge-m3"}'`
   → 返回 1024 维向量，且 L2 范数 ≈ 1.0。
6. **OOM 演练**：临时把 `MEM_FRACTION` 调到 0.35 触发 OOM → 验证服务**不崩**、
   自动降 batch，日志打印降级信息。
7. **长文本**：提交一段 ≥ 8192 token 文本 → 校验截断/分块行为与显存表现。
8. **落盘**：`.gitignore` 生效，`models/` 未被纳入版本管理；Conventional Commits 提交成功。

## 八、风险与回退

- **6GB 显存 + 桌面共用**：若实测稳态占用逼近上限，回退措施依次为 → 降低 `max_length` 至 512 →
  降 batch 至 4 → 提高 `MEM_FRACTION` 冗余（调低至 0.55）→ 长文本改 CPU 处理。
- **FlagEmbedding 依赖冲突**：回退方案为 `transformers` 直调 + CLS pooling + L2 归一化（等价实现）。
- **WDDM 换页**：硬上限已阻断；若仍出现卡顿，说明桌面占用过高，走体检提示清理流程。
