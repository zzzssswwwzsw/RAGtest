# bge-m3 本地嵌入服务（6GB 显存 / CUDA）

在 6GB 显存的移动端显卡上常驻运行 [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
稠密向量模型，对外提供 OpenAI 兼容的 `/v1/embeddings` 接口。

设计前提只有一条：**宁可降级、变慢，也不能把桌面卡死或让服务崩掉。**

- 稠密向量 1024 维，已 L2 归一化，可直接用内积当余弦相似度
- FP16 推理，进程级显存硬上限 + token 预算组批
- 真 OOM 时逐级降级（清缓存 → 批减半 → 单条转 CPU），服务本身不崩
- 加载前显存体检：不足时列出占用进程由你决定清理谁，**绝不自动结束他人进程**

本 README 面向**部署与运维**；**调用方**（写代码调接口的人）请看
[USER_MANUAL.md](USER_MANUAL.md)，那份侧重接口契约、批量与超时策略、向量使用与错误处理。

设计依据、取舍过程与实测记录见 [bge-m3-cuda-embedding-plan.md](bge-m3-cuda-embedding-plan.md)。

## 目录结构

```
src/settings.py     显存预算常量、档位预设、环境变量装配
src/preflight.py    体检：显存/内存探测、占用进程列表、交互降级、CLI 入口
src/embedder.py     编码器：组批、FP16 前向、OOM 降级链路
src/api.py          FastAPI：/v1/embeddings、/healthz、/preflight
scripts/start.ps1   启动脚本：探端口 → 体检 → 单 worker 拉起服务
scripts/bench.py    显存阶梯压测，用于回填档位预估值
tests/test_smoke.py 冒烟测试（模型用例默认跳过）
```

## 环境准备

### 1. 依赖

```powershell
conda create -n bgem3 python=3.10 -y
conda activate bgem3
pip install -r requirements.txt
```

`requirements.txt` 里 torch 走 PyTorch 的 cu121 专用源：**公共 PyPI 的 Windows 构建不带 CUDA**，
直装会静默退化成 CPU 推理（能跑，但慢两个数量级）。

### 2. 模型权重

首次启动会自动下载到 `models/`（约 2.1 GiB）。也可事先手动放好 `models/bge-m3/`。
来源按 `BGE_MODEL_ID` 环境变量 > 本地 `models/bge-m3/` > Hub 仓库名 顺序解析，因此放好本地副本即可离线启动。

> 不要设置 `HF_ENDPOINT` 指向 hf-mirror：它对 LFS 大文件返回 308 且丢掉 `X-Repo-Commit`，
> 而 `huggingface_hub` 不跟跨域跳转，表现为「小文件成功、大文件失败」。详见
> [scripts/start.ps1](scripts/start.ps1) 顶部注释。

### 3. 本机私有配置

```powershell
cp .env.example .env.local
```

`.env.local` 已 gitignore，用来放本机绝对路径：

```
BGE_PYTHON=D:\path\to\conda\envs\bgem3\python.exe
```

## 启动

```powershell
powershell -File scripts/start.ps1
```

脚本按顺序做四件事，任一步失败即停：

1. 解析解释器（`-Python` > `.env.local` 的 `BGE_PYTHON` > PATH），并校验它能 import 项目依赖
2. 探测端口是否可用——uvicorn 在模型加载完之后才 bind，先探能省下一次无用的模型加载
3. 显存体检，把最终档位写入 `.env.runtime` 由服务继承
4. 以**单 worker** 启动 uvicorn（多 worker 会在显存里复制多份模型）

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `-Force` | 关 | 不交互，显存不足时直接降档继续 |
| `-Port` | 8000 | 监听端口 |
| `-BindHost` | 127.0.0.1 | 默认只对本机开放 |
| `-Python` | 空 | 解释器路径，覆盖 `.env.local` |
| `-EnvOut` | .env.runtime | 档位交接文件 |

服务起来后 `http://127.0.0.1:8000/docs` 是交互式接口文档。

## 接口

### `POST /v1/embeddings`

```powershell
curl -s -X POST http://127.0.0.1:8000/v1/embeddings `
  -H "Content-Type: application/json" `
  -d '{"input": "测试文本", "model": "bge-m3"}'
```

```json
{
  "object": "list",
  "data": [{"object": "embedding", "index": 0, "embedding": [0.0123, -0.0456, "... 共 1024 维"]}],
  "model": "bge-m3",
  "usage": {"prompt_tokens": 13, "total_tokens": 13}
}
```

- `input` 支持单条字符串或字符串数组，单请求上限 256 条
- `encoding_format` 支持 `float`（默认）与 `base64`（1024 × float32 原始字节的 base64）
- `max_length` 可选，不得超过当前档位的序列上限
- `usage` 是所有输入的 token 总数，按实际分词计算并计入截断

### `GET /healthz`

存活探针，附带当前档位、后端与显存占用，例如：

```json
{
  "status": "ok",
  "model": "D:\\...\\models\\bge-m3",
  "backend": "flagembedding",
  "preset": "safe",
  "max_length": 512,
  "batch": 8,
  "token_budget": 8192,
  "memory": {"device": "cuda", "limit_mib": 3809, "allocated_mib": 1085, "reserved_mib": 1090, "peak_mib": 1085},
  "degradations": []
}
```

`degradations` 非空说明有请求被兜底到 CPU（运行期降级），`preset` 带 `(forced)` 后缀说明启动时体检已降档。

### `GET /preflight`

运行期显存体检，随时可查还剩多少余量：`ok`、`need_mib`、`free_mib`、`host_ram_ok`，
以及显存总量与占用进程列表。

### 错误码

| 状态码 | 含义 |
| --- | --- |
| 400 | 入参不合法：条数超限、`input` 为空、`encoding_format` 不支持、`dimensions` 非 1024、`max_length` 超档位 |
| 422 | 请求体不合 schema：`input` 不是字符串或字符串数组，或字段越界（如 `max_length=0`） |
| 503 | 模型尚未就绪（启动中） |
| 507 | 显存不足且逐级降级后仍失败，请缩短文本或降低并发 |

## 配置项

均为环境变量，非法值直接报错退出，不带着错配置启动。

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `BGE_PRESET` | safe | 档位：safe / balanced / long |
| `BGE_MAX_LENGTH` | 档位值 | 覆盖序列上限 |
| `BGE_BATCH` | 档位值 | 覆盖批大小 |
| `BGE_MEM_FRACTION` | 0.62 | 进程级显存上限占显卡总量的比例 |
| `BGE_TOKEN_BUDGET` | 8192 | 单批 `批大小 × 序列长度` 上限 |
| `BGE_CONCURRENCY` | 2 | 同时进行的编码请求数（信号量） |
| `BGE_DEVICE` | cuda | cuda / cpu |
| `BGE_BACKEND` | auto | auto / flagembedding / transformers |
| `BGE_MODEL_ID` | 空 | 覆盖模型来源（路径或 Hub 仓库名） |
| `BGE_FORCE` | 空 | 置 1 等同 `-Force` |

`BGE_PYTHON` 与 `BGE_HF_HOME` 只被 [scripts/start.ps1](scripts/start.ps1) 读取，放在 `.env.local`。

## 档位与实测数据

`python scripts/bench.py` 在本机（RTX 3060 Laptop 6GB / torch 2.3.1+cu121）实测：

| 序列长度 | 批 | 单批 token | 峰值显存 | 耗时 | 说明 |
| --- | --- | --- | --- | --- | --- |
| 512 | 8 | 4096 | 1277 MiB | 1.50 s | ← safe 档（默认） |
| 1024 | 8 | 8192 | 1717 MiB | 0.50 s | ← balanced 档 |
| 2048 | 4 | 8192 | 2217 MiB | 0.76 s | |
| 4096 | 2 | 8192 | 3217 MiB | 1.25 s | long 档取 batch=1（峰值约 2154 MiB） |
| 8192 | 1 | 8192 | — | 153 s | **不可用**：必然 OOM 后转 CPU |

峰值含权重（约 1090 MiB）。8192 档之所以不作为可选档位，是因为它在本卡上必然 OOM，
随后静默兜底到 CPU（单条 153 s），`bench.py` 会把这种「看起来通过」的档位标出来并排除出推荐表。

换机器或换 torch 版本后应重跑 `bench.py`，再把实测值回填到 [src/settings.py](src/settings.py) 的 `PRESETS`。

## 测试

```powershell
python -m pytest tests/ -q                      # 纯逻辑：组批、配置装配、体检
$env:BGE_SMOKE_MODEL=1; python -m pytest tests/ -q   # 追加需要真实权重与显存的用例
```

## 已知限制

- **WDDM 不上报单进程显存占用**，体检的进程表只能给名字（会打印提示），仍需人工判断谁是显存大户
- **真正先咬住的是 token 预算**，不是显存上限：默认预算把单批激活压到约 200 MiB，
  硬上限主要在超长单序列时兜底。反过来，`BGE_TOKEN_BUDGET` 调大后压力会转移到硬上限
- 体检的 `need_mib` 是偏保守的规划估值（weight + 上下文 + 激活 + 800 MiB 余量）。
  桌面负载较高时会自动降档到 `safe(forced)` batch=1，日志与 `/healthz` 都会显式标明
- **CPU 兜底很慢**：4096 token 单条约 45 s。客户端超时设置要据此放宽，否则会看到超时而不是结果；
  兜底还要在宿主内存里再放一份 fp32 权重（约 2.3 GiB），`/preflight` 的 `host_ram_ok` 为 false 时表示不一定给得起
- 仅在 Windows 11 + NVIDIA + CUDA 环境验证过；非 Windows 下控制台编码相关的兜底逻辑不会生效

## 排障

| 现象 | 原因与处理 |
| --- | --- |
| `ERROR: port 8000 ... already in use` | 端口被占，用 `-Port` 换端口或停掉旧进程 |
| 启动时 `exit 3` / `No module named uvicorn` | `.env.local` 的 `BGE_PYTHON` 未配置或指向了不含项目依赖的环境 |
| `/healthz` 显示 `preset: safe(forced)` | 启动时体检判定显存不足并自动降档；看启动日志里「空闲 X < 所需 Y」那行，清理占用进程后重启 |
| 请求返回 507 | 显存不足且降级失败，缩短文本或降低并发 |
| 日志出现「已转 CPU 兜底处理」 | 该请求没跑在 GPU 上，慢是正常的，`/healthz` 的 `degradations` 也会记录 |
| 日志里体检报告位置/顺序异常 | 重定向输出时为行缓冲，不会滞留；若整段缺失说明进程被强杀 |
