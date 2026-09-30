# rag-demo

最小可运行的本地 RAG（检索增强生成）练手项目，全程离线、无需 API Key、clone 即跑。

包含两个渐进式 Demo，外加一个检索优化实验：

| 文件 | 技术栈 | 生成方式 | 定位 |
|---|---|---|---|
| `rag_demo.py` | Chroma + ONNX MiniLM | 无 LLM，仅检索展示 | 理解向量检索环节 |
| `rag_demo_llamaindex.py` | LlamaIndex + bge-small-zh + llama.cpp (Qwen2.5) | 本地 LLM 生成完整回答 | 端到端 RAG 链路 |
| `rag_retrieval_lab.py` | LlamaIndex + bge-reranker + BM25 | 无 LLM，纯检索评估 | 量化对比检索/重排/混合策略 |

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
| 关键词检索 | 无 | `BM25Retriever`（bm25s，需自定义中文分词，见踩坑 9） |
| 重排 Rerank | 无 | `SentenceTransformerRerank`（bge-reranker-base，见踩坑 10） |

## 快速开始

### 前置要求

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)（包管理器）

### 1. 安装依赖

```bash
uv sync
# 这条命令会装齐所有依赖，含 llama-cpp-python 的预编译 wheel。
# 它在 PyPI 只有源码包（Windows 直装会触发本地 C++ 编译），
# pyproject.toml 已把官方 wheel 源绑定到该包，原理见踩坑 7。
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

# 实验：检索优化对比（纯检索，不加载 LLM，秒级完成）
uv run python rag_retrieval_lab.py
```

首次运行 Demo 2 会自动下载 bge-small-zh 向量模型（约 100MB）；之后索引持久化，秒级启动。
实验脚本自带语料、建到独立索引目录（不复用 Demo 2 的库），首次运行会下载 bge-reranker-base（约 278MB）并向量化 56 条语料。

## 检索优化实验

`rag_retrieval_lab.py` 自带一份约 56 条、覆盖 10 个主题簇的语料（故意塞入大量"硬负例"——与答案共享词汇但并非答案的文档），用带 ground truth 的评测集量化对比四种检索配置。全程不加载 LLM，秒级跑完。语料与索引目录（`./data/llamaindex_lab_store`）独立于两个 Demo，互不影响。

### 四种配置

| 配置 | 检索链路 |
|---|---|
| A. baseline | 向量检索 Top-3（Demo 2 方案，对照组） |
| B. +rerank | 向量 Top-10 → bge-reranker 重排 → Top-3 |
| C. +hybrid | 向量 Top-10 + BM25 Top-10 → RRF 融合 → Top-3 |
| D. hybrid+rerank | 向量 + BM25 → RRF 融合 Top-10 → reranker 重排 → Top-3 |

### 实测结果（56 条语料，23 条正样本查询）

| 配置 | Hit@3 | Recall@3 | MRR@3 |
|---|---|---|---|
| A. baseline | 95.7% | 85.1% | 0.891 |
| B. +rerank | 100.0% | 93.1% | 0.971 |
| C. +hybrid | 95.7% | 88.8% | 0.891 |
| D. hybrid+rerank | 100.0% | 93.1% | 0.971 |

### 关键结论

1. **reranker 是最大功臣**：Hit@3 95.7%→100%、Recall@3 85.1%→93.1%、MRR@3 0.891→0.971。典型例子"本地没有GPU怎么跑大模型"（答案 `llama.cpp`）：向量检索被"本地/大模型"带偏到模型本身（Llama3、Qwen），完全漏掉；reranker 用 cross-encoder 精读 query 与文档，理解了"怎么跑"指向推理引擎，把它从榜外捞回第 1 位。
2. **纯 BM25 混合检索收益有限**：Recall 仅 +3.6%，Hit/MRR 不变。因为中文 query 多是语义改写（"没有GPU跑大模型" vs 文档里的"C++实现的LLM推理引擎"），字面词几乎不重叠，BM25 抓不到。它的价值在专有名词、精确术语匹配的场景。
3. **hybrid+rerank 与单独 rerank 打平**：reranker 足够强时，初检多召回的候选被它重新排序吸收，混合检索的边际贡献消失。说明工程上"向量 Top-N + reranker"往往就是性价比最高的组合。
4. **数据量决定能否看出差异**：上一版只有 8 条时 Hit/Recall 全饱和在 100%，只能靠 MRR 挤牙膏；扩到 56 条并加入硬负例后，差距才真实显现。检索实验的信噪比，首先取决于语料规模与负例质量。

> 设计约束：评测集里每条 expected 数量都不超过 Top-3 坑位数——期望数一旦超过 K，Recall@k 就有结构性天花板（如期望 4 篇时最高 0.75），那是指标失真而非检索失败。校验闸门会强制拦截这类条目。

### 评测集设计：内置手写为基准，RAGAS 生成仅为实验路径

`rag_retrieval_lab.py` 的评测集经历了一次方向修正（过程见踩坑 15）：**内置 `EVAL_SET` 是唯一可信基准**，RAGAS 自动生成降级为实验路径。

内置评测集的设计原则：

1. **人工逐条校订 ground truth**——自动标签实测 30/30 系统性错标，评测集宁可少而准（现 28 正 + 3 负）；
2. **每条带 type 标注，每类查询压测一种检索机制**：`keyword`（精确术语，BM25 应占优）/ `paraphrase`（语义改写，向量应占优）/ `interference`（词面干扰硬负例，reranker 应占优）/ `multi-doc`（聚合对比）/ `factual`（基础盘）/ `negative`（库外问题，考察误命中）。总体指标打平时，分类型报表仍能定位各配置在哪类查询上占优；
3. **expected 数量 ≤ TOP_K**——否则 Recall@3 被结构性封顶（期望 4 篇最高只能 0.75），指标失真而非检索差；
4. **加载闸门**：内置与外部 evalset.json 一律过 `validate_eval_set()` 校验（未知 doc_id / 超过 TOP_K / 空 query 直接剔除并红字报错），杜绝坏标签静默产出假指标；
5. **复现零 API 依赖**：`git clone` → `uv sync` → `uv run python rag_retrieval_lab.py`，全程不出网（模型走本地缓存/镜像）。

RAGAS 路径（`gen_evalset.py`）保留作实验，但其产出必须人工校订后才可入评测集。

### 自动生成评测集（RAGAS + GLM，⚠️ 实验路径，产出需人工校订）

手写 `EVAL_SET` 的瓶颈在规模：语料一扩充，人工标注跟不上。`gen_evalset.py` 用 [RAGAS](https://docs.ragas.io) 从 CORPUS 自动生成问答评测集：

```powershell
# 推荐：项目根建 .env 文件（已被 .gitignore 忽略），内容一行：
#   ZHIPUAI_API_KEY=你的key
# 然后直接运行：
uv run python gen_evalset.py            # 默认生成 30 题，产出 evalset.json
uv run python rag_retrieval_lab.py      # 自动检测并加载 evalset.json（过校验闸门）
```

> ⚠️ 别用 `set ZHIPUAI_API_KEY=xxx`：在 PowerShell 里 `set` 是 `Set-Variable` 别名，只建了脚本内变量，不会成为环境变量，子进程读不到（实测踩坑，见第 12 条）。临时设置要用 `$env:ZHIPUAI_API_KEY="xxx"`，且只对当前终端会话生效。

> ⚠️ 实测该路径产出的评测集**不能直接用**：ground truth 曾 30/30 错标、23/30 题目是英文（模板语言所致），详见踩坑 15。lab 的校验闸门只能拦截"引用不存在的 doc_id"这类硬错误，**拦不住语义层面的错标**。

分工设计（对应"出题用强模型、跑实验用本地模型"的原则）：

| 环节 | 模型 | 位置 |
|---|---|---|
| 出题（知识图谱抽取、问题进化） | glm-4-flash（智谱 API，可用 `GLM_MODEL` 换） | 云端，一次性 |
| 问题去重/聚类 | bge-small-zh | 本地，复用已有 |
| 日常检索实验 | 无 LLM（纯检索） | 全本地 |

生成结果固化成 `evalset.json` 后，日常实验不再碰 API。`reference_contexts` 通过原文子串匹配映射回 `doc_id`（收集全部匹配，不做 first-hit 短路——踩坑 15 的事故根源就是这里短路），映射失败的条目直接丢弃。负样本（知识库无答案的问题）RAGAS 不产，继续手工维护在 `NEGATIVE_QUERIES`。

## 配置项（环境变量）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `CHROMA_PATH` | `./data/chroma_db` | Chroma 持久化目录 |
| `LLAMAINDEX_PATH` | `./data/llamaindex_store` | LlamaIndex 索引持久化目录 |
| `LLAMAINDEX_LAB_PATH` | `./data/llamaindex_lab_store` | 检索实验的独立索引目录 |
| `MODEL_PATH` | `./models/qwen2.5-1.5b-instruct-q4_k_m.gguf` | GGUF 模型路径 |
| `MODEL_DIR` | `./models` | download_model.py 的下载目录 |
| `HF_ENDPOINT` | 无 | 国内网络设为 `https://hf-mirror.com` 加速 HuggingFace 下载 |
| `ZHIPUAI_API_KEY` | 无 | 智谱开放平台 API key，仅 gen_evalset.py 出题时需要 |
| `GLM_MODEL` | `glm-4.5-air` | 出题模型 id，以智谱开放平台实际可用名为准 |
| `GLM_MAX_TOKENS` | `8192` | 出题模型输出上限，须为整数；GLM-4.5-Air 是推理模型，thinking token 会吃掉输出预算，给少了会因 `finish_reason=length` 报错 |
| `HF_HUB_OFFLINE` | 无 | 设为 `1` 跳过 HF 联网检查，模型已缓存时离线秒启动 |

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

同类陷阱：**collection 的距离空间（`hnsw:space`）一旦创建就不能改**。Chroma 默认是 L2（欧氏距离）而不是余弦；本项目曾漏设该参数，却按"余弦距离"去换算，`1 - dist` 打出了 `相似度: -8.13%` 这种负值（检索本身没坏，是显示与注释错了）。现在 `rag_demo.py` 显式声明 `hnsw:space=cosine`，并在检测到沿用旧集合时给出提示——删除 `data/chroma_db` 重建即可。

### 5. Windows 装 llama-cpp-python 用预编译 wheel

直接 `pip install llama-cpp-python` 在 Windows 上会触发本地 C++ 编译，极易失败。官方提供了各平台预编译 wheel，本项目**已在 `pyproject.toml` 里声明专属 wheel 源**（写法见踩坑 7），`uv sync` 会自动取到，无需手工操作。

临时手工安装（不推荐：不属于依赖声明，下次 `uv sync` 会被清掉）：

```bash
uv pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

GPU 版把 URL 里的 `cpu` 换成 `cu121` 等（需与本机 CUDA 版本匹配）。

### 6. LlamaIndex 必须显式设置 `Settings.llm`

LlamaIndex 默认 LLM 是 OpenAI，不设置就在构建索引/查询时尝试联网调用 OpenAI API。纯本地场景要么置 `Settings.llm = None`（纯检索），要么设为 `LlamaCPP(...)`。

### 7. 依赖必须写进 `pyproject.toml`，`uv pip install` 装的不算数

`uv pip install` 不写入 `pyproject.toml`/`uv.lock`，之后执行 `uv sync` 会把它装的包**从环境中删掉**。本项目真实踩到过：`llama-index-llms-llama-cpp` 曾靠 `uv pip install` 安装，一次 `uv sync` 后就变成 `ModuleNotFoundError: No module named 'llama_index.llms'`。

正确做法是把依赖声明在 `pyproject.toml` 里。`llama-cpp-python` 的特殊之处是 PyPI 只有源码包，Windows 直装会触发本地 C++ 编译且极易失败；官方提供各平台预编译 wheel，用 uv 的"专属源"声明即可让 `uv sync` 自动取到 wheel，不必手工安装：

```toml
[[tool.uv.index]]
name = "llama-cpp-cpu"
url = "https://abetlen.github.io/llama-cpp-python/whl/cpu"
explicit = true          # 只服务显式绑定的包，不影响其它依赖取源

[tool.uv.sources]
llama-cpp-python = { index = "llama-cpp-cpu" }
```

两个配套细节：

- 该源对 `cp313-win_amd64` 目前只出到 `0.3.19`，所以 `pyproject.toml` 给 `llama-cpp-python` 加了 `<0.3.20` 上限。**没有上限时 uv 会挑到更高版本，但那只在 PyPI 有源码包，又会回到本地编译**——这个坑我们实际撞了一次才发现。
- `uv.lock` 必须提交（已从 `.gitignore` 移除），否则 clone 后解析出的版本组合不可复现，见踩坑 11。

### 8. 国内网络：HuggingFace 下载加镜像

```bash
set HF_ENDPOINT=https://hf-mirror.com
```

对 bge 向量模型和 GGUF 模型下载都有效（download_model.py 内部已默认设置）。

### 9. BM25 中文检索必须自定义分词，否则得分全为 0

`BM25Retriever` 默认 `token_pattern=r"(?u)\b\w\w+\b"`。中文字符属于 `\w` 且词间无空格，整句会被切成**单个 token**，中文查询几乎无法匹配，实测 BM25 得分全部为 0。

解决：传"中文按单字 + 英文按词"的正则，并跳过英文词干化：

```python
BM25Retriever.from_defaults(
    index=index,
    token_pattern=r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+",  # 中文单字 + 英文词
    skip_stemming=True,                              # 中文场景 stemming 无意义
)
```

单字切分对中文召回足够（BM25 靠多字重合打分），若要更精准可上 jieba 分词，但会新增依赖。

### 10. reranker 输出的是 logits，不是概率

`SentenceTransformerRerank` 用的 bge-reranker 输出范围约 -10~+10（cross-encoder logits），**不是** 0~1 的相似度。直接按 `{score:.2%}` 打印会出现"350%"这类荒谬值。

处理：分数只用于**排序**；需要展示"置信度"时套一层 sigmoid 归一化。它与向量检索的余弦相似度不可直接比较。

### 11. ragas 0.4.3 必须配 langchain 0.3.x，装最新会 ImportError

`import ragas` 时报 `No module named 'langchain_community.chat_models.vertexai'`——ragas 0.4.3 裸依赖 `langchain-community`（无上限），uv 拉了 0.4.2，而该模块在 0.4.0 被移除。

```toml
# ✅ 锁 0.3.x 组合
"ragas>=0.4.3,<0.5"
"langchain-community>=0.3.0,<0.4"
"langchain-openai>=0.3.0,<0.4"
"langchain-huggingface>=0.3.0,<0.4"
```

0.3.x 依赖 `langchain-core<1.0`，所以 langchain-openai/huggingface 也要同步降到 0.3 系列，不能只降 community。这套只服务 `gen_evalset.py`，lab 和两个 demo 走 llama_index，不受牵连。

### 12. PowerShell 里 `set KEY=value` 不会设置环境变量

`set` 在 cmd 里是设环境变量，但在 PowerShell 里是 `Set-Variable` 的别名——只创建脚本内变量，**不会传给子进程**。表现为明明执行了 `set ZHIPUAI_API_KEY=xxx`，python 里 `os.getenv` 依然拿到 None。

```powershell
set ZHIPUAI_API_KEY=xxx              # ❌ cmd 语法，PowerShell 下无效
$env:ZHIPUAI_API_KEY = "xxx"         # ✅ 当前会话生效
# ✅✅ 推荐：项目根 .env 文件（gitignore 已排除），gen_evalset.py 自动加载
```

### 13. LlamaIndex 与 langchain 的模型缓存目录不同，离线模式直接崩

`gen_evalset.py` 启动后先卡在反复 `WinError 10060` 重试（1s→2s→4s→8s→8s），加 `HF_HUB_OFFLINE=1` 后又变成 `LocalEntryNotFoundError` 直接报错。两个现象同一根因：**模型缓存位置不一致**——

- LlamaIndex 的 `HuggingFaceEmbedding` 把模型下到 `%LOCALAPPDATA%\llama_index\...\Cache`
- langchain 的 `HuggingFaceEmbeddings` 只查标准 HF 缓存（`HF_HOME` 或 `~/.cache/huggingface`）

模型其实早已缓存，但 langchain 在标准位置找不到：联网时它去 hub 重下（表现为 retry 卡顿），离线时直接报错。

解法：`gen_evalset.py` 里的 `resolve_local_model()` 搜遍两处缓存，命中就把 snapshot 本地路径直接传给 embedding，绕开 hub 名称查找。`HF_HUB_OFFLINE=1` 可继续保留（跳过联网检查，秒启动）。

> 附带一个日志误导点：`LLM is explicitly disabled. Using MockLLM.` 是 `import rag_retrieval_lab` 时其模块级 `Settings.llm = None` 打的，属检索实验专用；出题的 GLM 在 `generate()` 里单独创建，两者无关。看到这条不代表云端模型没生效。

### 14. 靠 prompt 让 1.5B 模型"不知道就说不知道"没用，要在检索侧挡噪声

Demo 2 里 `如何给Agent添加长期记忆？` 检索是对的（命中文档 68.56%，与次高分断层明显），但 1.5B 模型仍编造："在CrewAI中，role、goal、backstory 这三种元素共同构成了Agent的长期记忆"——原文完全没有这句话。加 prompt 约束后连续两版都失败，且失败模式会**摆动**：

| 版本 | 判据 | 结果 |
|---|---|---|
| v1 | "话题相近但不含答案就拒答" | **过度拒答**：知识库明明有答案（问"如何添加"，文档讲"分两类"），也判成"话题相近" |
| v2 | 收紧到"完全不同的主题"才拒答 | 拒答没了，但**编造回来了**，且换了形式 |

结论：1.5B 对多规则指令的遵循已到天花板，继续加规则是递减收益。**有效杠杆在检索侧**——让模型压根看不到噪声。

做法（`rag_demo_llamaindex.py` 的 `SCORE_GAP_RATIO`）：只保留相似度 ≥ 最高分 × 0.85 的片段。注意这里**不能用绝对阈值**：

| 查询 | 正解分 | 次高分 | 正解/噪声比值 |
|---|---|---|---|
| LangGraph和CrewAI有什么区别 | 50.01% | 47.15% | **0.94**（正解，必须保留） |
| 如何给Agent添加长期记忆 | 68.56% | 51.75% | 0.76（噪声，应滤掉） |
| 企业级Agent架构怎么设计 | 71.55% | 57.87% | 0.81（噪声，应滤掉） |
| 本地运行的轻量模型有哪些 | 60.82% | 47.53% | 0.78（噪声，应滤掉） |

正解分在 44%~72% 之间浮动，**任何绝对阈值都会误杀**（例如取 0.55 会把第一题的正解全部滤掉，直接答不出）。而"与最佳命中差多少"这个相对关系稳定：有效区间是 (0.81, 0.94]，故取 0.85。

实测效果：四题全部改为精准作答，编造与无关片段罗列同时消失（Q1 仍保留 3 条，因为它本就需要多文档）。**代价与局限要讲清楚**：0.85 是在这 4 条样本上标定的启发式值，不是验证过的最优解；语料或 Embedding 模型一换就需重新标定，并且它只对"单主题查询"有效——真正的多主题聚合查询会被它误伤。

### 15. RAGAS 自动生成的评测集 30/30 错标：ground truth 必须人工校订

按"出题用强模型"的思路接入 RAGAS + GLM 后，生成流程顺利跑通（换 `glm-4-flash` 后 30 题 0 丢弃、约 4 分钟），但产出的评测集**不能直接用**，两个致命问题：

**问题一：ground truth 30/30 系统性错标（映射代码 bug，可修）。** 现象：问 CrewAI 的题被标到 `d_langgraph`，问 Milvus/HNSW 被标到 `d_chroma`——全部 30 条都错标成"该主题在 CORPUS 里的第一条"。根因是 `contexts_to_doc_ids` 里命中一条就 `break`：RAGAS 返回的 reference_contexts 是整篇合并文档（含主题下每一句原文），本应收集全部匹配得到整组 member_id，一旦 first-hit 短路就塌缩成第一条。修复为收集全部匹配。

**问题二：23/30 的题目是英文（模板性缺陷，难修）。** 语料是中文、Embedding 是 bge-small-zh、BM25 是中文单字分词，英文 query 打中文语料得到的指标没有意义。根因是 RAGAS 的 synthesizer prompt 模板是英文的，模型照模板出题；要出中文题需自定义中文 prompt，属于改 RAGAS 内部行为。另外 `multi_hop_abstract` 类问题语义牵强（"FastAPI 如何应对短期记忆挑战"这类概念嫁接），质量明显低于单跳题。

**更深一层的教训**：校验闸门（查 doc_id 是否存在）拦得住"引用不存在的 id"，**拦不住"标签语义就错了"**——错标的 id 都真实存在，只是张冠李戴。自动生成评测集的瓶颈从来不是生成，是**校订成本**；没有人工校订环节的自动标签，规模越大污染越快。

**最终取向**：评测集回到人工设计（内置 `EVAL_SET`，带 type 分类标注，见"评测集设计"一节）；RAGAS 路径保留作实验，产出必须人工校订后才能使用。复现实验的路径因此变成零 API 依赖的 `git clone` → `uv sync` → `uv run`。

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
├── rag_retrieval_lab.py      # 实验: 检索优化对比（reranker + BM25）
├── gen_evalset.py            # 评测集生成: RAGAS + GLM 云端出题 → evalset.json
├── download_model.py         # GGUF 模型下载脚本
├── pyproject.toml            # 项目元数据与依赖（uv 管理）
├── uv.lock                   # 锁定依赖版本（必须提交，保证 clone 后可复现）
├── .gitignore
├── data/                     # 向量库持久化（git 忽略，运行时生成）
└── models/                   # GGUF 模型文件（git 忽略）
```
