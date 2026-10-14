# bge-m3 嵌入服务 · 用户手册

**这份手册写给调用方**：你写代码调这个服务，需要知道接口契约、怎么发批量、超时怎么设、
向量怎么用、出错怎么办。你不需要关心显存。

装服务、启服务、调档位、排障请看 [README.md](README.md)；
本次所有数字都在本机（RTX 3060 Laptop 6GB）实测得到，换机器会变。

服务默认监听 `http://127.0.0.1:8000`，下文用 `BASE` 代指。

## 目录

- [1. 五分钟上手](#1-五分钟上手)
- [2. 接口契约](#2-接口契约)
- [3. 客户端写法](#3-客户端写法)
- [4. 检索示例](#4-检索示例)
- [5. 运行期观测](#5-运行期观测)
- [6. 常见问题](#6-常见问题)
- [7. 兼容性边界](#7-兼容性边界)

## 1. 五分钟上手

```bash
curl -s -X POST http://127.0.0.1:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "测试文本", "model": "bge-m3"}'
```

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.0123, -0.0456, "...共 1024 个数"]}
  ],
  "model": "bge-m3",
  "usage": {"prompt_tokens": 13, "total_tokens": 13}
}
```

三件事确认服务可用：`status` 是 200、`embedding` 长度 1024、`usage.prompt_tokens` 大于 0。

Python 最小客户端：

```python
import requests

BASE = "http://127.0.0.1:8000"


def embed(texts: list[str], timeout: float = 120) -> list[list[float]]:
    """把一批文本编码成 1024 维归一化向量，返回顺序与输入一一对应。"""
    response = requests.post(
        f"{BASE}/v1/embeddings",
        json={"input": texts, "model": "bge-m3"},
        timeout=timeout,
    )
    response.raise_for_status()
    return [item["embedding"] for item in response.json()["data"]]
```

`timeout` 别用默认值（`None` = 永不超时），也别设得太小，理由见
[3.3 超时与重试](#33-超时与重试)。

## 2. 接口契约

### 2.1 请求字段 `POST /v1/embeddings`

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `input` | string \| string[] | 是 | 单条或一批，**单请求最多 256 条**；只能是字符串，不支持 token id 数组 |
| `model` | string | 否 | 默认 `bge-m3`。**不做校验，原样回显**，见下方提示 |
| `encoding_format` | `float` \| `base64` | 否 | 默认 `float`；`base64` 是 1024 个 float32 的原始字节，省带宽 |
| `max_length` | int ≥ 1 | 否 | 本次请求的序列上限，不得超过当前档位上限（默认档位是 512） |
| `dimensions` | int | 否 | 只接受 1024 或省略；本模型不支持降维 |

未知字段（如 OpenAI SDK 会带的 `user`）一律被忽略，不会报错。

> **`model` 不校验**：写成 `text-embedding-3-small` 甚至空串都返回 200，并把你的字符串原样回显。
> 也就是说，如果你沿用别处的代码、只把 `base_url` 指过来，向量会是 bge-m3 的 1024 维而不报错。
> 请自行确认客户端里写的模型名与实际服务一致。

### 2.2 响应字段

| 字段 | 说明 |
| --- | --- |
| `object` | 固定 `list` |
| `data[].index` | 该向量对应输入的下标，**顺序与输入严格一致**（内部会重排组批，但结果已还原） |
| `data[].embedding` | 1024 个数；`encoding_format=float` 时是数组，`base64` 时是字符串 |
| `model` | 回显请求里的 `model` |
| `usage.prompt_tokens` | 所有输入的 token 总数，按实际分词计算并计入截断 |

### 2.3 语义细节

**向量已 L2 归一化**，范数为 1（实测 0.999858 ~ 1.000196）。因此：

- **内积即余弦相似度**，不需要再算模长；
- **不要再归一化一次**，重复归一化在数值上无害但纯属浪费。

**截断与 `usage` 的关系**：文本超过 `max_length` 会被截断（只编码前 `max_length` 个 token，
不报错）。判断是否被截断的实用办法：

> 若 `usage.prompt_tokens` 正好等于当前的 `max_length`，基本可以断定这条被截断了。

当前档位上限随时可用 [`/healthz`](#5-运行期观测) 查到（`max_length` 字段）。

**`float` 与 `base64` 的体积**（实测，含 JSON 包装）：

| 编码 | 1 条向量 | 8 条向量 |
| --- | --- | --- |
| `float` | 19488 字节 | 155591 字节 |
| `base64` | 5600 字节 | 约 45 KB |

`base64` 约省 3.5 倍带宽，且逐位可逆（实测与 `float` 解出的 float32 数组完全一致）：

```python
import base64
import numpy as np

payload = response.json()["data"][0]["embedding"]      # encoding_format=base64
vector = np.frombuffer(base64.b64decode(payload), dtype=np.float32)   # shape (1024,)
```

**空串是个坑**：`input=""` 或纯空白会返回 200 和一个范数为 1 的向量——它只编码了特殊 token，
与任何文本都谈不上语义相似。这类向量混进向量库会污染召回。

> 请在客户端侧过滤空串与纯空白：`texts = [t.strip() for t in texts if t.strip()]`。

### 2.4 错误码与处置

| 码 | 含义 | 你该做什么 |
| --- | --- | --- |
| 400 | 入参不合法：条数 > 256、`input` 为空列表、`encoding_format` 不是 float/base64、`dimensions` 不是 1024、`max_length` 超过档位上限 | 改请求；`detail` 里有具体原因 |
| 422 | 不符合 schema：`input` 不是字符串/字符串数组（如传了数字或 token 数组）、`max_length=0` 等字段越界 | 改请求 |
| 503 | 模型尚未就绪（服务刚启动、正在加载权重） | 退避重试；`/healthz` 能通了再发业务请求 |
| 507 | 显存不足，且逐级降级后仍失败 | **不要盲目重试**：缩短文本、减小批量，或查看 [`/preflight`](#5-运行期观测) 后清理显存占用 |

## 3. 客户端写法

### 3.1 批量：一次发多少条

服务端会自己按 token 预算组批，你不需要按档位去凑批大小。实践建议：

- **别一条一条发**。单条请求约 0.14s，但其中有固定开销；一次发几十条能显著摊薄（实测 32 条 / 8416 token 只要 0.47s）。
- **上限 256 条**，实测 256 条短文本（9216 token）1.10s，完全可以一次发满。
- **长文本要按 token 估**：批量耗时由 `条数 × 截断后长度` 决定。发 256 条 1000 字的文本，
  截断到 512 后总量约 13 万 token，耗时约 8~10 秒。
- **要算总 token 预算**：`sum(min(len(每条), 档位上限))` 除以 8192 就是大致批次数，乘以约 0.5s 就是量级。

### 3.2 延迟量级（实测，默认档位）

| 请求 | token | 耗时 |
| --- | --- | --- |
| 1 条短文本 | 5 | 0.14 s |
| 32 条 × 约 400 字 | 8416 | 0.47 s |
| 256 条 × 约 60 字 | 9216 | 1.10 s |
| 256 条 × 约 1000 字（截断到 512） | 约 131000 | 约 8~10 s |

以上都是 GPU 路径。**一旦某条请求被降级到 CPU，单条 4096 token 要 45 秒**，见
[6. 常见问题](#6-常见问题)。

### 3.3 超时与重试

**关键事实（实测）：客户端超时不会取消服务端的计算。**

客户端在 1.5s 放弃等待后，那个大请求仍在服务端跑完并正常返回（服务端日志记录了 200），
并且一直占着一个并发槽。也就是说，"超时后立刻重试"只会让 GPU 上叠加更多计算，越试越慢。

据此建议：

- `timeout` 设 **120 秒**（而不是 5 或 10 秒）；批量越大越要放宽。
- 超时后**不要立即重试**，也不要并发重试；先确认服务还活着（`/healthz`），再决定是减小批量还是等。
- 按错误码区分重试与否：`503` 退避重试；`507` 减小负载；`400/422` 改请求；**5xx 之外不要重试**。

### 3.4 并发

服务端默认只允许 **2 个** 编码请求同时进行（`BGE_CONCURRENCY`），多余的会在信号量上排队。

- 客户端并发 **2~4 路** 足够打满；开到几十路只会让所有请求一起变慢。
- 同一个服务是**单 GPU 共享资源**，多个进程/多个同事同时调用时要约定用量，别互相拖慢。

### 3.5 向量落库

- 维度 **1024**，类型 **float32**（`base64` 直接解出 float32；`float` 路径建议落库前 `np.float32`）。
- 相似度度量选 **cosine**（等价地：对归一化向量用 inner product）。不要选欧氏距离。
- 不要重复归一化（已归一化）。
- 向量里**不含** query/document 的类型信息：本服务不对文本做任何前缀或指令加工，
  你只要保证 query 和 document 走同一个接口编码即可。

## 4. 检索示例

完整可运行（下面是实测输出）：

```python
import numpy as np
import requests

BASE = "http://127.0.0.1:8000"

docs = [
    "bge-m3 支持稠密、稀疏与多向量三种检索方式",
    "今天食堂的红烧肉有点咸",
    "Windows 下 PowerShell 读取无 BOM 的 UTF-8 文件会按 ANSI 解码",
    "向量数据库通常用余弦相似度或内积做召回",
]
query = "怎么把文本变成向量做召回"

response = requests.post(
    f"{BASE}/v1/embeddings",
    json={"input": docs + [query], "model": "bge-m3"},
    timeout=120,
)
vectors = np.asarray([d["embedding"] for d in response.json()["data"]], dtype=np.float32)

# 已 L2 归一化，内积即余弦
scores = vectors[:-1] @ vectors[-1]
for rank, i in enumerate(np.argsort(-scores), start=1):
    print(f"{rank}. {scores[i]:+.4f}  {docs[i]}")
```

实测排序与得分：

```
1. +0.7447  向量数据库通常用余弦相似度或内积做召回
2. +0.4986  bge-m3 支持稠密、稀疏与多向量三种检索方式
3. +0.4961  Windows 下 PowerShell 读取无 BOM 的 UTF-8 文件会按 ANSI 解码
4. +0.3762  今天食堂的红烧肉有点咸
```

注意得分的绝对水平（0.38~0.74）：bge-m3 的相似度**普遍偏高**，两条无关文本也可能有 0.3 以上。
**不要用 0.8 之类的固定阈值做"是否相关"的判断**，用排序取 Top-K，阈值请在你自己数据上标定。

## 5. 运行期观测

### `GET /healthz`

```json
{
  "status": "ok",
  "preset": "safe",
  "max_length": 512,
  "batch": 8,
  "token_budget": 8192,
  "degradations": []
}
```

调用方需要关注三个字段：

| 字段 | 看点 |
| --- | --- |
| `max_length` | 当前档位的序列上限，决定你的文本会不会被截断 |
| `degradations` | 非空表示有请求被兜底到 CPU（慢的元凶就在这），数组里会写明是哪条文本 |
| `preset` | 带 `(forced)` 后缀表示启动时显存不足已自动降档，你能用的上限随之变小 |

### `GET /preflight`

显存体检：`ok`、`need_mib`、`free_mib`、`host_ram_ok`，以及占用显存的进程列表。
收到 `507` 时看它，判断是"现在显存被别的程序占了"还是"你这次批量太大"。

## 6. 常见问题

**Q：这条请求为什么花了 45 秒？**
该请求被降级到了 CPU。触发条件是 GPU 显存装不下这个批次（通常是单条超长文本）。查
`/healthz` 的 `degradations`，非空即命中。要避免就缩短单条文本、减小批量。

**Q：返回 507，我的载荷不大啊？**
507 表示显存不足且降级也失败，通常是**别的程序占着显存**。用 `/preflight` 看空闲显存与占用进程，
清理后重试；同时减小批量。

**Q：长文本会报错吗？**
不会，会被静默截断到档位上限（默认 512 token，约 512 个汉字）。用
`usage.prompt_tokens == max_length` 判断截断。要更长文本，需要在部署侧换档位（见 README）。

**Q：向量看起来都差不多 / 相似度普遍很高？**
先排除空串（见 [2.3](#23-语义细节)）。其次注意 bge-m3 的相似度天然偏高，用排序而非绝对阈值。

**Q：能换模型或降维吗（`dimensions=512`）？**
不能，固定 bge-m3 稠密向量 1024 维；`dimensions` 只接受 1024 或省略。

**Q：模型名写错了为什么没报错？**
`model` 只回显不校验，见 [2.1](#21-请求字段-post-v1embeddings)。

**Q：连接被拒绝 / 连不上？**
服务没起来或换了端口。启动与排障见 [README.md](README.md)。

**Q：`input` 能传 token 数组吗？**
不能，只接受字符串或字符串数组，传数字数组会 422。

## 7. 兼容性边界

本服务实现的是 OpenAI `/v1/embeddings` 的**子集**，与官方 API 的差异：

| 能力 | 本服务 |
| --- | --- |
| 未知字段（`user` 等） | 接受并忽略（实测 200） |
| `model` 取值 | 不校验，原样回显 |
| `input` 为 token id 数组 | **不支持**（422） |
| `dimensions` 降维 | **不支持**（只接受 1024） |
| `encoding_format` | 只支持 `float`、`base64`（`hex` 等 → 400） |
| 单请求条数上限 | 256（超出 → 400） |
| 输出维度 | 固定 1024，已 L2 归一化 |
| 稀疏 / ColBERT 多向量 | **不提供**，只有稠密向量 |
| 鉴权 | 无（默认只绑定 127.0.0.1） |

因此把既有客户端的 `base_url` 指过来通常可直接跑通，但要注意 `dimensions` 与 token 数组这两项会直接报错，
以及 `model` 不校验这一静默风险。
