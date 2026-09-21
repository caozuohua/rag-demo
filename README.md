# rag-demo

最小可运行的本地 RAG（检索增强生成）练手项目，全程离线、无需 API Key、clone 即跑。

包含两个渐进式 Demo：

| 文件 | 技术栈 | 生成方式 | 定位 |
|---|---|---|---|
| `rag_demo.py` | Chroma + ONNX MiniLM | 无 LLM，仅检索展示 | 理解向量检索环节 |
| `rag_demo_llamaindex.py` | LlamaIndex + bge-small-zh + llama.cpp (Qwen2.5) | 本地 LLM 生成完整回答 | 端到端 RAG 链路 |

## 核心理念

### RAG 三步曲

```
离线索引：文档 → Embedding → 向量库
在线检索：用户问题 → Embedding → 相似度匹配 → Top-K 命中文档
增强生成：命中文档拼成上下文 → LLM 依据上下文回答
```

### 端到端链路（LlamaIndex 版）

```
用户问题
   ↓
bge-small-zh-v1.5 (100MB, CPU) 将问题向量化
   ↓
VectorStoreIndex 检索 Top-3（持久化于 ./data/llamaindex_store）
   ↓
命中文档拼进 prompt，套 Qwen ChatML 模板
   ↓
llama.cpp 加载 Qwen2.5-1.5B-Instruct GGUF (1GB, CPU) 生成回答
```

### 组件对应关系（Chroma 版 → LlamaIndex 版）

| 概念 | Chroma 版 | LlamaIndex 版 |
|---|---|---|
| 向量库落盘 | `PersistentClient(path)` | `storage_context.persist()` |
| 写入文档 | `collection.add(ids, documents, metadatas)` | `VectorStoreIndex.from_documents()` |
| 语义检索 | `collection.query(query_texts)` | `index.as_retriever().retrieve()` |
| 相似度语义 | `distances` 余弦**距离**（越小越好，`1-dist` 转相似度） | `node.score` 相似**度**（越大越好） |
| Embedding | `DefaultEmbeddingFunction`（MiniLM，英文为主） | `HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")`（中文优化） |
| LLM | 无 | `LlamaCPP`（llama.cpp 本地推理） |

## 快速开始

### 前置要求

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)（包管理器）

### 1. 安装依赖

```bash
uv sync
# llama-cpp-python 装预编译 wheel，避免 Windows 本地编译（为什么见"踩坑经验"）
uv pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
uv pip install llama-index-llms-llama-cpp
```

### 2. 下载 LLM 模型（约 1GB）

```bash
set HF_ENDPOINT=https://hf-mirror.com
uv run python download_model.py
```

### 3. 运行

```bash
# Demo 1：纯向量检索（Chroma）
uv run python rag_demo.py

# Demo 2：端到端 RAG（LlamaIndex + 本地 LLM，CPU 推理每题约 10-30 秒）
uv run python rag_demo_llamaindex.py
```

首次运行 Demo 2 会自动下载 bge-small-zh 向量模型（约 100MB）；之后索引持久化，秒级启动。

## 配置项（环境变量）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `CHROMA_PATH` | `./data/chroma_db` | Chroma 持久化目录 |
| `LLAMAINDEX_PATH` | `./data/llamaindex_store` | LlamaIndex 索引持久化目录 |
| `MODEL_PATH` | `./models/qwen2.5-1.5b-instruct-q4_k_m.gguf` | GGUF 模型路径 |
| `MODEL_DIR` | `./models` | download_model.py 的下载目录 |
| `HF_ENDPOINT` | 无 | 国内网络设为 `https://hf-mirror.com` 加速 HuggingFace 下载 |

## 踩坑经验

均为本项目实战中真实踩到的坑。

### 1. llama.cpp 的惩罚参数叫 `repeat_penalty`，不是 `repetition_penalty`

```python
generate_kwargs={"repeat_penalty": 1.1}   # ✅ llama.cpp 原生命名
generate_kwargs={"repetition_penalty": 1.1}  # ❌ HF Transformers 命名，报错
```

LlamaCPP 集成层不做参数名映射，直接透传给 llama-cpp-python。

### 2. Instruct 模型必须套 ChatML 模板，裸 prompt 会复读循环

直接把拼好的纯文本喂给 `llm.complete()`，Qwen 会复读 prompt 指令字面、陷入循环式重复。必须包进 ChatML 结构：

```
<|im_start|>system
{系统指令}<|im_end|>
<|im_start|>user
{检索知识 + 问题}<|im_end|>
<|im_start|>assistant
```

这是修复后回答质量显著提升的最关键一步。

### 3. 小模型（≤2B）温度建议降到 0.3

0.7 在大模型上合理，但在 1.5B 模型上会放大发散与循环倾向。搭配 `repeat_penalty=1.1` 使用。

### 4. 换 Embedding 模型必须删掉旧索引重建

不同 Embedding 模型的向量空间互不兼容。换模型后继续用旧索引会导致检索结果错乱（且不报错，更隐蔽）。删除 `data/llamaindex_store/` 重新构建即可。

### 5. Windows 装 llama-cpp-python 用预编译 wheel

直接 `pip install llama-cpp-python` 在 Windows 上会触发本地 C++ 编译，极易失败。官方提供了各平台预编译 wheel：

```bash
uv pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

GPU 版把 `cpu` 换成 `cu121` 等（需与本机 CUDA 版本匹配）。

### 6. LlamaIndex 必须显式设置 `Settings.llm`

LlamaIndex 默认 LLM 是 OpenAI，不设置就在构建索引/查询时尝试联网调用 OpenAI API。纯本地场景要么置 `Settings.llm = None`（纯检索），要么设为 `LlamaCPP(...)`。

### 7. `uv pip install` 与 `uv add` 不要混用

`uv pip install` 不写入 `pyproject.toml`/`uv.lock`，之后执行 `uv sync` 会把未锁定的包**从环境中删掉**。本项目现状：`llama-cpp-python` 因需要额外 wheel 源，用 `uv pip` 安装；每次 `uv sync` 后需重跑上面第 1 步的两条 `uv pip install` 命令。

### 8. 国内网络：HuggingFace 下载加镜像

```bash
set HF_ENDPOINT=https://hf-mirror.com
```

对 bge 向量模型和 GGUF 模型下载都有效（download_model.py 内部已默认设置）。

## 常见问题

**Q：检索相似度只有 60-70%，是不是太低？**
bge 的余弦相似度分布本身偏低，0.6+ 对中文短文本已是较强匹配。判断检索质量看"命中文档是否切题"，比看绝对数值更可靠。

**Q：LLM 回答质量不够好？**
1.5B 模型能力有限，属预期。可换 `Qwen2.5-3B-Instruct`（约 2GB）或 7B（约 4.7GB），改 `download_model.py` 的 `MODEL_REPO/MODEL_FILE`，再设 `MODEL_PATH` 指向新文件即可，代码无需改动。

**Q：想换别的 Embedding？**
改 `HuggingFaceEmbedding(model_name="...")` 一行，推荐候选：`BAAI/bge-large-zh-v1.5`（质量更好、CPU 较慢）、`moka-ai/m3e-base`。换完务必见踩坑经验第 4 条。

## 目录结构

```
rag-demo/
├── rag_demo.py               # Demo 1: Chroma 纯检索
├── rag_demo_llamaindex.py    # Demo 2: LlamaIndex 端到端 RAG
├── download_model.py         # GGUF 模型下载脚本
├── pyproject.toml            # 项目元数据与依赖（uv 管理）
├── .gitignore
├── data/                     # 向量库持久化（git 忽略，运行时生成）
└── models/                   # GGUF 模型文件（git 忽略）
```
