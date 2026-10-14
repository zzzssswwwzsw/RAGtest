# bge-m3 嵌入服务 · 脚本用户手册

**这份手册写给操作本项目脚本的人**：把服务跑起来、压测档位、验收接口、跑测试。

- 想**写代码调接口**（请求/响应字段、批量、超时、向量怎么用）→ 见 [README.md](README.md) 的「接口」一节
- 想了解**为什么这么设计**（显存预算、降级链路、实测过程）→ 见 [bge-m3-cuda-embedding-plan.md](bge-m3-cuda-embedding-plan.md)

所有数字均在本机（RTX 3060 Laptop 6GB / torch 2.3.1+cu121）实测，换机器会变。

## 目录

- [1. 脚本总览](#1-脚本总览)
- [2. 一次性准备](#2-一次性准备)
- [3. 启动服务 start.ps1](#3-启动服务-startps1)
- [4. 单独体检 preflight](#4-单独体检-preflight)
- [5. 验收 check_api.py](#5-验收-check_apipy)
- [6. 压测与回填档位 bench.py](#6-压测与回填档位-benchpy)
- [7. 测试 pytest](#7-测试-pytest)
- [8. 典型工作流](#8-典型工作流)
- [9. 退出码与排障速查](#9-退出码与排障速查)

## 1. 脚本总览

| 脚本 | 干什么 | 要服务已启动吗 | 什么时候用 |
| --- | --- | --- | --- |
| `scripts/start.ps1` | 探端口 → 体检 → 单 worker 拉起服务 | — | 每次开发 |
| `python -m src.preflight` | 只体检、只列占用进程，不启动服务 | 否 | 启动被挡下时 |
| `scripts/check_api.py` | 黑盒验收：接口契约 + 检索质量 | **是** | 改配置 / 换机器 / 部署后 |
| `scripts/bench.py` | 逐档实测峰值显存，产出档位建议 | 否，自己加载模型 | 换机器 / 换 torch 版本 |
| `pytest tests/` | 单元测试与冒烟测试 | 否 | 改动前后 |

> **`bench.py` 和 `preflight` 都要自己加载模型**，所以**不能在服务运行期间跑**：
> 6GB 卡装不下两份权重，会互相挤爆。先停服务再跑。

## 2. 一次性准备

```powershell
conda create -n bgem3 python=3.10 -y
conda activate bgem3
pip install -r requirements.txt
```

再把解释器路径写进本机私有配置（已 gitignore）：

```powershell
cp .env.example .env.local
# 编辑 .env.local：BGE_PYTHON=D:\path\to\conda\envs\bgem3\python.exe
```

`start.ps1` 解析解释器的优先级：`-Python` 参数 > `.env.local` 的 `BGE_PYTHON` > 同名环境变量 > PATH 上的 `python`。
落到 PATH 兜底时它会先 `import fastapi, uvicorn` 验一次，失败就 `exit 3`，不会让你等到体检跑完才报错。

## 3. 启动服务 start.ps1

```powershell
powershell -File scripts/start.ps1
```

脚本会先 `Set-Location` 到仓库根，所以**在哪个目录调用都行**。

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `-Force` | 关 | 非交互，显存不足时直接降档继续（等价于 `BGE_FORCE=1`） |
| `-Port` | 8000 | 监听端口 |
| `-BindHost` | 127.0.0.1 | 默认只对本机开放 |
| `-Python` | 空 | 解释器路径，覆盖 `.env.local` |
| `-EnvOut` | `.env.runtime` | 档位交接文件 |

### 四步流程

1. **解析解释器**并验证依赖可导入
2. **探端口** —— 用 `TcpListener` + `ExclusiveAddressUse` 试占一下。这步不能省：
   uvicorn 要等 FastAPI 的 lifespan（含模型加载）跑完才 bind，端口冲突会先白烧十几秒和约 1.5 GiB 显存
3. **显存体检**，把定下的档位写进 `.env.runtime`
4. **单 worker 拉起 uvicorn** —— 多 worker 会在显存里复制多份模型，6GB 卡直接爆

### 日志怎么读

```
[ OK ] 显存体检通过，开始加载模型
  档位 safe  序列上限 512  批大小 8  硬上限 3809 MiB
  显存总量 6144 MiB  [########------------------------]  已用 1549  空闲 4594  (来源 nvml)
  本次所需 2700 MiB = 权重 1150 + 上下文 500 + 激活 250 + 余量 800
  系统内存可用 5505 MiB（CPU 兜底门槛 4096）[OK]
  占用显存的进程（按占用降序）：
         PID  进程                                   显存
       32340  TextInputHost.exe                   未上报
    提示：WDDM 驱动未上报单进程占用，需按进程名人工判断。
已写出 .env.runtime：BGE_PRESET=safe BGE_MAX_LENGTH=512 BGE_BATCH=8
  inherited BGE_PRESET=safe from .env.runtime
Starting bge-m3 embedding service on http://127.0.0.1:8000 (single worker)...
INFO:     Waiting for application startup.
INFO bge.embedder: 显存硬上限已设为 3809 MiB（总量 6143 × 0.62）
INFO bge.embedder: 模型 D:\path\to\models\bge-m3 已加载（后端 flagembedding，设备 cuda，档位 safe(override)）
INFO bge.api: 服务就绪，接口见 /docs、/healthz、/preflight
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

两个正常但容易误会的点：

- **体检报告打印了两遍**（`safe` 和 `safe(override)`）。第一遍是启动脚本的命令行体检，
  第二遍是服务自己启动时又查一次当保险；显示 `(override)` 是因为档位改由 `.env.runtime` 显式指定。
- **`Uvicorn running on ...` 之前连不上是正常的**。要等模型加载完才 bind，约十几到几十秒。

停服务：在跑它的窗口按 `Ctrl+C`。

## 4. 单独体检 preflight

只想看显存够不够、谁占着，不想加载模型：

```powershell
python -m src.preflight                 # 交互模式
python -m src.preflight --force         # 非交互，不足时自动降档
python -m src.preflight --env-out .env.runtime
```

| 参数 | 说明 |
| --- | --- |
| `--force` | 非交互模式，显存不足时自动降档继续 |
| `--env-out PATH` | 把最终档位写成 env 文件（默认 `.env.runtime`） |

**交互模式**下显存不足会列出占用进程并让你决定清理谁，**绝不自动结束他人进程**。
写出的 `.env.runtime` 含四个键（`BGE_PRESET` / `BGE_MAX_LENGTH` / `BGE_BATCH` / `BGE_MEM_FRACTION`），
`start.ps1` 会逐行继承，保证体检降过的档不会在服务侧被还原回去。

## 5. 验收 check_api.py

```powershell
python scripts/check_api.py
python scripts/check_api.py --base http://127.0.0.1:8000 --timeout 120
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--base` | `http://127.0.0.1:8000` | 服务地址 |
| `--timeout` | 120.0 | 单请求超时秒数 |

只通过 HTTP 打接口、不 import 仓库任何模块，因此**可以直接拷给调用方**。四组共 17 项断言：

| 组 | 检查什么 |
| --- | --- |
| 服务状态 | `/healthz` 可达、`degradations` 为空 |
| 向量化契约 | 形状 1024、`index` 顺序、L2 归一化、单条与批量方向一致、乱序还原、可复现性 |
| 编码与边界 | `base64` 与 `float` 逐位一致、超长截断、256 / 257 条上限、空串行为 |
| 检索质量 | 三组标注语料（含期望命中文档），验证目标文档确实排 Top1 |

实测输出（默认档位，耗时 1.82s）：

```
── 2/4 向量化契约 ──
  [PASS] 输出形状 (n, 1024)  实测 (3, 1024)
  [PASS] index 与输入顺序一一对应  index=[0, 1, 2]
  [PASS] 向量已 L2 归一化  范数 0.999850 ~ 1.000202
  [PASS] 单条与批量的方向一致  余弦 1.000435，分量偏差 2.44e-04（FP16 批效应，正常）
  [PASS] 乱序输入能按 index 正确还原  偏差 0.00e+00 / 0.00e+00
  [PASS] 同一输入、同批量组成下逐位可复现  两次偏差 0.00e+00

── 3/4 编码与边界 ──
  [PASS] base64 与 float 逐位一致  dtype=float32 shape=(1024,)
  [PASS] base64 体积更小  float 19512 B -> base64 5600 B
  [PASS] 超长文本截断到档位上限  16384 字符 -> prompt_tokens=512（档位上限 512）
  [PASS] 256 条可接受  HTTP 200
  [PASS] 257 条被拒 (400)  HTTP 400 单次请求最多 256 条文本，收到 257 条
  [WARN] 空串返回可用向量（已知行为）  空串 L2=1.0002，与正常文本相似度 +0.5087

── 4/4 检索质量 ──
  [PASS] 检索命中「怎么把文本变成向量做召回」  Top1=doc0  doc0:+0.6686  doc3:+0.4914 ...

结论：全部通过（17 项），警告 1 项，耗时 1.82s
```

`[WARN]` 不算失败，意思是「服务行为符合预期，但你要知道这件事」。退出码 `0` 全部通过、`1` 有失败项，
可直接接进部署后的验收步骤。

其中「单条与批量的方向一致」这项断言的是**方向**而不是逐位相等，原因是实测发现：批大小一变，
FP16 的 reduction 顺序就变，同长同内容的三条批里就有 `2.44e-04` 的分量偏差（与 padding、批内位置都无关），
方向偏差则在 `1e-5` 量级、不影响排序。要做向量指纹或逐位比对，必须固定批量组成。

## 6. 压测与回填档位 bench.py

```powershell
python scripts/bench.py                      # 默认阶梯 512/1024/2048/4096/8192
python scripts/bench.py --lengths 512,1024   # 只跑指定档
python scripts/bench.py --batch 4            # 强制批大小，覆盖按预算推导的值
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--lengths` | `512,1024,2048,4096,8192` | 逗号分隔的序列长度阶梯 |
| `--out` | `.bench_result.json` | 结果落盘路径（已 gitignore） |
| `--batch` | 0 = 按 token 预算推导 | 强制指定批大小 |

**跑之前先停服务**（理由见 [1. 脚本总览](#1-脚本总览)）。输出形如：

```
  len   512  batch  8    1.500s  峰值  1277 MiB  设备增量   190 MiB
  len  1024  batch  8    0.500s  峰值  1717 MiB  设备增量   630 MiB
  len  4096  batch  2    1.250s  峰值  3217 MiB  设备增量  2130 MiB
  len  8192  batch  1  153.000s  峰值    ... MiB  设备增量    ... MiB  (已降级到 CPU，本档 GPU 不可行)
```

两种标记要看懂：

- **`已降批`**：这一档按预算推导的批大小触发了 OOM，脚本自动减半重试后成功
- **`已降级到 CPU，本档 GPU 不可行`**：编码器内部 OOM 后自己降到了 CPU，这一档**根本没跑在 GPU 上**。
  不标出来就会被误判成"通过"，然后推荐一个不可行的档位 —— 8192 档就是这样（单批 153s）

结果写入 `.bench_result.json`，同时打印建议回填的档位：

```
建议写回 settings.PRESETS 的档位（activation_mib 已含 20% 余量）：
  safe: {'max_length': 512, 'batch': 8, 'activation_mib': 224}
  ...
```

把 `max_length` / `batch` / `activation_mib` 填进 [src/settings.py](src/settings.py) 的 `PRESETS`。

## 7. 测试 pytest

```powershell
python -m pytest tests/ -q                            # 纯逻辑：组批、配置装配、体检
$env:BGE_SMOKE_MODEL=1; python -m pytest tests/ -q    # 追加需要真实权重与显存的用例
```

默认跑的用例不需要模型和显存，改配置逻辑后随手跑一遍即可。
带 `BGE_SMOKE_MODEL=1` 的那些会真的加载模型，**同样要先停服务**。

## 8. 典型工作流

### 首次部署

```powershell
python -m src.preflight            # 先单独看一眼显存够不够，不够就清进程
powershell -File scripts/start.ps1 # 起服务，等 Uvicorn running
python scripts/check_api.py        # 另开一个窗口验收，17 项全过就算可用
```

### 换机器 / 换 torch 版本

档位预估值是按卡实测的，换环境必须重测：

```powershell
python scripts/bench.py            # 停服务后跑
# 把打印的建议回填 src/settings.py 的 PRESETS
powershell -File scripts/start.ps1
python scripts/check_api.py
```

### 排查「变慢了」

```
请求变慢
  ├─ 查 /healthz 的 degradations  → 非空：有请求掉到 CPU 兜底了（单条 4096 token 约 45s）
  │                                 处理：缩短单条文本 / 减小批量
  ├─ 查 /healthz 的 preset        → 带 (forced)：启动时就降档了，上限比预期小
  └─ 查 /preflight                → 看空闲显存和占用进程，判断是别人抢了显存还是你批次太大
```

## 9. 退出码与排障速查

| 来源 | 码 | 含义 |
| --- | --- | --- |
| `start.ps1` | 1 | 端口被占；用 `-Port` 换端口或停掉旧进程 |
| `start.ps1` | 3 | 解释器不能导入服务依赖；配 `.env.local` 的 `BGE_PYTHON` |
| `src.preflight` | 1 | 显存体检未通过，服务未启动 |
| `src.preflight` | 2 | 环境变量配置非法（未知档位等），拒绝带错配置启动 |
| `bench.py` | 2 | 未启用 CUDA，压测无意义 |
| `check_api.py` | 0 / 1 | 全部通过 / 有失败项 |

| 现象 | 处理 |
| --- | --- |
| `port 8000 ... already in use` | `-Port 8001` 换端口，或停掉旧进程 |
| 启动时 `exit 3` / `No module named uvicorn` | `.env.local` 的 `BGE_PYTHON` 没配或指向了不含依赖的环境 |
| `/healthz` 显示 `preset: safe(forced)` | 启动时体检判定显存不足并自动降档；看启动日志「空闲 X < 所需 Y」那行，清理占用进程后重启 |
| 日志出现「已转 CPU 兜底处理」 | 该请求没跑在 GPU 上，慢是正常的；`/healthz` 的 `degradations` 也会记录 |
| 体检进程表全是「未上报」 | WDDM 驱动不上报单进程显存，只能按进程名人工判断谁是显存大户 |
| `check_api.py` 报连不上 | 服务没起或端口不对；`Uvicorn running on ...` 那行才是就绪信号 |
| `check_api.py` 某项 FAIL | 先看该行右侧的实测值；`[WARN]` 不是失败项 |
| 日志里体检报告位置/顺序异常 | 重定向输出时为行缓冲，不会滞留；若整段缺失说明进程被强杀 |
