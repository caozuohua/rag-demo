#!/usr/bin/env python3
"""
RAG 检索优化实验 - reranker + BM25 混合检索

纯检索评估脚本：不加载 LLM、不生成答案，只对比四种检索配置的排序质量。

四种配置：
  A. baseline      向量检索 Top-3
  B. +rerank       向量检索 Top-N -> bge-reranker 重排 -> Top-3
  C. +hybrid       向量 Top-N + BM25 Top-N -> RRF 融合 -> Top-3
  D. hybrid+rerank 向量 + BM25 -> RRF 融合 Top-N -> reranker 重排 -> Top-3

指标（对每个查询算，再取宏平均）：
  Hit@k     前 k 是否至少命中一个相关文档
  Recall@k  前 k 覆盖了多少比例的相关文档
  MRR@k     第一个命中的倒数排名（衡量"命中得早不早"）

与 rag_demo_llamaindex.py 的关系：
  本实验自带一份约 56 条的更大语料（含大量"硬负例"——与答案共享词汇
  但并非答案的文档），建到独立的索引目录，不复用也不影响 demo 的 8 条。
  候选池远大于 Top-K 后，Recall 才有下降空间，四种配置的差距才显现。

评测集：内置 EVAL_SET（人工逐条校订、带 type 分类标注），是唯一可信基准；
  evalset.json（RAGAS 自动生成）存在时会被加载，但必须通过校验闸门——
  自动标签曾 30/30 系统性错标，见 README 踩坑 15。

运行：
  set HF_ENDPOINT=https://hf-mirror.com
  uv run python rag_retrieval_lab.py            # 四配置对比
  uv run python rag_retrieval_lab.py --sweep    # 附加：阈值敏感性实验（标定 SCORE_GAP_RATIO）
首次运行会下载 bge-reranker-base（约 278MB）并向量化语料。
"""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever

# 纯检索实验，显式禁用 LLM：QueryFusionRetriever 在 num_queries=1 时
# 不会调用 LLM，但仍会 resolve 一个 Settings.llm 占位（MockLLM，零成本）
# 注意：embed_model 较重（需加载 bge 模型），放到 main() 里才设置，
# 保证本模块可被 gen_evalset.py 轻量 import（只取 CORPUS/EVAL_SET）
Settings.llm = None

# 独立索引目录，与 demo 的 llamaindex_store 完全隔离
PERSIST_DIR = os.getenv("LLAMAINDEX_LAB_PATH", "./data/llamaindex_lab_store")

# 评测集文件：由 gen_evalset.py 产出。锚定项目根，不依赖运行时 CWD
EVALSET_PATH = Path(__file__).resolve().parent / "evalset.json"

# BM25 中文分词方案（实测结论，见 README 踩坑 9）：
# 默认 pattern 会把整句中文切成单 token，导致中文查询 BM25 得分全为 0。
# 这里改成"中文按单字 + 英文按词"切分，并跳过英文词干化。
BM25_TOKEN_PATTERN = r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+"

# reranker：cross-encoder，输出 logits（约 -10~+10），不是概率，仅用于排序
RERANKER_MODEL = "BAAI/bge-reranker-base"

TOP_K = 3        # 最终评估的坑位数
CANDIDATE_K = 10 # 交给重排/融合的候选池大小（远小于语料总数，才有区分度）

# 阈值敏感性实验（--sweep）的扫描网格：
# 相对比例 = Demo 2 的 SCORE_GAP_RATIO 思路（保留 score >= ratio * top1），
#            当时只在 4 条样本上标定，这里放到全套评测集上验证；
# 绝对门槛 = top1 分数低于 floor 时整题拒答（相对比例对负样本无效，
#            负样本的分数同样抱团，只有绝对门槛才可能拒掉库外问题）。
RELATIVE_RATIOS = [0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
ABSOLUTE_FLOORS = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

# ============================================================
# 语料库（约 56 条，覆盖 10 个主题簇，故意制造跨簇/簇内硬负例）
# 每条：id / topic（展示用）/ text
# ============================================================

CORPUS = [
    # --- Agent 编排框架 ---
    ("d_langgraph", "Agent框架", "LangGraph是基于图结构的Agent编排框架，核心概念包括State（状态）、Node（节点）、Edge（边）。State是整个工作流的共享记忆。"),
    ("d_crewai", "Agent框架", "CrewAI是角色驱动的多Agent框架，每个Agent有role、goal、backstory。支持sequential和hierarchical两种流程。"),
    ("d_autogen", "Agent框架", "AutoGen是微软推出的多Agent对话框架，Agent之间通过消息传递协作，支持人类参与和代码执行。"),
    ("d_semantickernel", "Agent框架", "Semantic Kernel是微软的轻量级Agent编排SDK，以插件和函数为核心，深度集成.NET生态。"),
    ("d_llamaindex", "Agent框架", "LlamaIndex是数据框架，核心是把外部数据接入LLM，提供索引、检索器、查询引擎，也支持Agent与工作流。"),
    ("d_langchain", "Agent框架", "LangChain是LLM应用开发框架，用链串联模型、工具、检索器，生态最全但抽象层次较多。"),
    # --- 推理算法 ---
    ("d_react", "推理算法", "ReAct算法：Reasoning加Acting循环。思考、行动、观察、再思考。适合探索性任务，灵活性高。"),
    ("d_planexecute", "推理算法", "Plan-and-Execute算法：先规划生成步骤列表，再逐步执行。适合长流程任务，效率高但灵活性低于ReAct。"),
    ("d_reflexion", "推理算法", "Reflexion通过自我反思改进：执行失败后生成语言化的反思记忆，指导下一次尝试，无需更新模型权重。"),
    ("d_tot", "推理算法", "Tree-of-Thought把推理组织成树，每个节点是一个中间想法，可展开、回溯，用BFS或DFS搜索最优路径。"),
    ("d_cot", "推理算法", "Chain-of-Thought提示让模型分步骤推理，通过示例或一步步想触发，提升复杂推理任务的准确率。"),
    # --- 向量数据库 ---
    ("d_chroma", "向量数据库", "Chroma是本地轻量的开源向量数据库，嵌入式部署，适合原型和小规模RAG，Python原生。"),
    ("d_milvus", "向量数据库", "Milvus是企业级分布式向量数据库，支持海量向量和HNSW、IVF等多种索引类型，需要独立部署。"),
    ("d_qdrant", "向量数据库", "Qdrant是用Rust编写的高性能向量数据库，支持丰富的过滤条件和量化，资源占用低。"),
    ("d_lancedb", "向量数据库", "LanceDB是嵌入式向量数据库，基于Lance列式格式，无需独立服务，直接读写本地文件。"),
    ("d_faiss", "向量数据库", "FAISS是Meta的向量检索库，不是数据库而是索引库，需自己管理持久化，检索速度极快。"),
    ("d_weaviate", "向量数据库", "Weaviate是带GraphQL的开源向量数据库，支持多租户、模块化向量化器和混合检索。"),
    ("d_pinecone", "向量数据库", "Pinecone是全托管的云向量数据库，免运维、支持大规模，按用量付费，非开源自部署。"),
    ("d_pgvector", "向量数据库", "pgvector是PostgreSQL的向量扩展，把向量存进关系表，可与业务数据一起用SQL查询。"),
    # --- Embedding 模型 ---
    ("d_bge_small", "Embedding", "bge-small-zh-v1.5是智源中文向量模型，约100MB，CPU可跑，适合中文短文本语义检索。"),
    ("d_bge_large", "Embedding", "bge-large-zh-v1.5是bge-large中文向量模型，约1.2GB，检索质量更高但CPU推理较慢。"),
    ("d_m3e", "Embedding", "m3e-base是Moka开源的中文通用向量模型，约500MB，在中文语义相似度任务上表现均衡。"),
    ("d_text2vec", "Embedding", "text2vec-chinese系列通过对比学习训练中文句向量，有bert-base等多档规格可选。"),
    ("d_e5", "Embedding", "E5是微软的多语言向量模型，查询和文档需加不同前缀query和passage以获得最佳效果。"),
    ("d_gte", "Embedding", "GTE是阿里的通用文本向量模型，有base和large多尺寸，MTEB榜单中文表现优秀。"),
    ("d_minilm", "Embedding", "all-MiniLM-L6-v2是sentence-transformers英文为主的轻量向量模型，约90MB，中文效果一般。"),
    # --- 本地推理 / 模型 ---
    ("d_qwen", "本地模型", "Qwen2.5是阿里开源大模型，有0.5B、1.5B、3B、7B等规格，中文优化好，提供Instruct和Base版本。"),
    ("d_llama3", "本地模型", "Llama3是Meta开源大模型，有8B和70B，英文能力强，中文需额外微调或选社区中文版。"),
    ("d_vllm", "推理引擎", "vLLM是高吞吐LLM推理引擎，用PagedAttention管理KV缓存，支持连续批处理，适合GPU服务化部署。"),
    ("d_llamacpp", "推理引擎", "llama.cpp是C++实现的LLM推理引擎，支持CPU和Metal，运行GGUF量化模型，适合本地无GPU部署。"),
    ("d_ollama", "推理引擎", "Ollama是本地大模型一键运行工具，封装llama.cpp，用Modelfile管理模型，提供REST API。"),
    ("d_gguf", "推理引擎", "GGUF是llama.cpp使用的模型量化格式，常见量化档Q4_K_M在体积和质量间取得平衡。"),
    ("d_kvcache", "推理引擎", "KV缓存缓存注意力层的Key和Value避免重复计算，长上下文时显存占用主要来自KV缓存。"),
    ("d_quantization", "推理引擎", "量化把模型权重从FP16降到INT8或INT4，显存和体积减半以上，代价是轻微精度损失。"),
    # --- RAG 技术 ---
    ("d_rag_flow", "RAG技术", "RAG检索增强生成流程：文档切分、Embedding、存入向量库、提问、检索TopK、LLM结合上下文生成，解决知识截止和幻觉。"),
    ("d_chunking", "RAG技术", "文本切块把长文档切成适合Embedding的片段，chunk_size和overlap是关键参数，影响检索粒度。"),
    ("d_rerank", "RAG技术", "重排用cross-encoder对初检TopN重新打分排序，比bi-encoder更准但更慢，常做两阶段检索的第二阶段。"),
    ("d_hybrid", "RAG技术", "混合检索结合向量语义检索和BM25关键词检索，用RRF等方式融合，兼顾语义和精确词匹配。"),
    ("d_graphrag", "RAG技术", "GraphRAG用LLM从文档抽取实体关系构建知识图谱，检索时沿图多跳，适合需要全局关联的问题。"),
    ("d_rageval", "RAG技术", "RAG评估常用指标：检索侧的Hit、Recall、MRR，生成侧的忠实度、相关性、答案正确性。"),
    ("d_hallucination", "RAG技术", "幻觉指LLM生成与事实或给定上下文不符的内容，RAG通过提供检索上下文可显著缓解。"),
    ("d_queryrewrite", "RAG技术", "查询改写用LLM把用户问题改写成多个变体或更规范的检索query，提升召回。"),
    # --- 记忆 ---
    ("d_shortmem", "记忆", "短期记忆通常指对话上下文窗口内的历史消息，随会话增长会溢出，需要截断或摘要。"),
    ("d_longmem", "记忆", "Agent长期记忆把重要信息向量化持久化到外部存储，跨会话检索召回，弥补上下文窗口有限。"),
    ("d_kgmem", "记忆", "知识图谱记忆用实体关系三元组组织Agent记忆，支持结构化查询和多跳关联。"),
    ("d_summarymem", "记忆", "对话摘要记忆把历史消息压缩成摘要再放入上下文，用更少token保留关键信息。"),
    # --- 提示 / 结构化输出 ---
    ("d_fewshot", "提示工程", "Few-shot提示在prompt里给几个输入输出示例，让模型模仿格式，无需训练即可引导行为。"),
    ("d_structured", "提示工程", "结构化输出约束模型按JSON Schema生成，便于下游程序解析，可用function calling或JSON mode实现。"),
    ("d_systemprompt", "提示工程", "系统提示定义模型的角色、规则和边界，优先级通常高于用户消息。"),
    # --- 微调 ---
    ("d_lora", "微调", "LoRA冻结原模型权重，只训练低秩旁路矩阵，显存和存储需求大幅下降，可插拔。"),
    ("d_qlora", "微调", "QLoRA在4位量化的基座模型上叠加LoRA，单卡即可微调大模型。"),
    ("d_fullft", "微调", "全参数微调更新所有权重，效果上限高但显存和算力成本最高，易在小数据上过拟合。"),
    ("d_dpo", "微调", "DPO直接偏好优化，用偏好对训练模型，比RLHF流程更简单稳定。"),
    # --- 部署 ---
    ("d_fastapi", "部署", "FastAPI是异步Python Web框架，常用于把LLM或RAG服务封装成REST接口，自动生成OpenAPI文档。"),
    ("d_streaming", "部署", "流式输出用SSE或WebSocket让LLM逐token返回，降低首字延迟，改善交互体验。"),
    ("d_docker", "部署", "Docker把应用和依赖打包成镜像，保证环境一致，是LLM服务部署的常见方式。"),
]

# ============================================================
# 评测集：内置、人工逐条校订，是检索质量的唯一可信基准
#
# 设计原则（为什么不用 RAGAS 自动生成，见 README 踩坑 15）：
#   1. ground truth 精确到 doc_id、人工校订——自动生成的标签实测
#      30/30 系统性错标，评测集宁可少而准
#   2. 每条带 type 标注，每类查询压测一种检索机制：
#        keyword      精确术语（PagedAttention/Q4_K_M/HNSW/RRF）—— BM25 应占优
#        paraphrase   语义改写（与答案几乎无词面重叠）—— 向量应占优
#        interference 词面干扰（同簇/跨簇硬负例密集）—— reranker 应占优
#        multi-doc    跨文档聚合对比 —— 考察 Top-K 覆盖
#        factual      单文档事实题 —— 基础盘
#        negative     负样本（库外无答案）—— 考察误命中
#   3. expected 数量 <= TOP_K：否则 Recall@3 被结构性封顶
#      （期望 4 篇时最高只能 0.75，指标失真而非检索差）
#   4. 复现零 API 依赖：clone -> uv sync -> uv run 本脚本
# ============================================================

EVAL_SET = [
    # --- keyword：精确术语查询（考察 BM25 精确词匹配的价值） ---
    ("PagedAttention解决了什么问题", ["d_vllm"], "keyword"),
    ("Q4_K_M是什么量化档位", ["d_gguf"], "keyword"),
    ("HNSW和IVF是哪类索引", ["d_milvus"], "keyword"),
    ("RRF是怎么融合两路检索结果的", ["d_hybrid"], "keyword"),
    # --- paraphrase：语义改写，与答案几乎无词面重叠（考察向量语义检索） ---
    ("本地没有GPU怎么跑大模型", ["d_llamacpp"], "paraphrase"),
    ("怎么评估检索质量", ["d_rageval"], "paraphrase"),
    ("两阶段检索的第二阶段是什么", ["d_rerank"], "paraphrase"),
    ("长文档怎么切分成片段", ["d_chunking"], "paraphrase"),
    ("对话历史太长装不下怎么办", ["d_summarymem", "d_shortmem"], "paraphrase"),
    ("怎么让模型按程序能解析的格式输出", ["d_structured"], "paraphrase"),
    ("希望Agent记住用户偏好、跨会话仍能想起，该怎么做", ["d_longmem"], "paraphrase"),
    # --- interference：词面干扰，同簇/跨簇硬负例密集（考察 reranker 排序纠偏） ---
    ("量化一定损失精度吗", ["d_quantization"], "interference"),
    ("KV缓存为什么会占大量显存", ["d_kvcache"], "interference"),
    ("AutoGen和Semantic Kernel分别是谁推出的", ["d_autogen", "d_semantickernel"], "interference"),
    ("bge-small-zh和m3e哪个更适合中文短文本", ["d_bge_small", "d_m3e"], "interference"),
    # --- multi-doc：跨文档聚合对比（考察 Top-K 覆盖能力） ---
    ("LangGraph和CrewAI有什么区别", ["d_langgraph", "d_crewai"], "multi-doc"),
    ("ReAct和Plan-and-Execute哪个更灵活", ["d_react", "d_planexecute"], "multi-doc"),
    ("LoRA和全参数微调的区别", ["d_lora", "d_fullft"], "multi-doc"),
    ("RAG能解决大模型的哪些问题", ["d_rag_flow", "d_hallucination"], "multi-doc"),
    # --- factual：单文档事实题（基础盘） ---
    ("AutoGen是什么框架", ["d_autogen"], "factual"),
    ("Qwen2.5有哪些规格", ["d_qwen"], "factual"),
    ("把RAG服务封装成HTTP接口用什么", ["d_fastapi"], "factual"),
    ("GraphRAG和普通RAG的区别", ["d_graphrag"], "factual"),
    ("怎么用一张消费级显卡微调大模型", ["d_qlora"], "factual"),
    ("few-shot提示是什么", ["d_fewshot"], "factual"),
    ("vLLM靠什么技术提高吞吐", ["d_vllm"], "factual"),
    ("智源开源的中文Embedding模型有哪些", ["d_bge_small", "d_bge_large"], "factual"),
    ("E5模型使用时要注意什么", ["d_e5"], "factual"),
    # --- negative：负样本，知识库无答案（考察各配置会不会误命中） ---
    ("今天天气怎么样", [], "negative"),
    ("推荐一部最近上映的电影", [], "negative"),
    ("红烧肉怎么做好吃", [], "negative"),
]


def _corpus_fingerprint() -> str:
    """语料内容指纹，用于判断索引是否需要重建。"""
    joined = "||".join(f"{i}:{t}" for i, _, t in CORPUS)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def _doc_id(node) -> str:
    """
    取节点对应的文档标识。

    node.id_ 是构建时生成的 UUID；真正的业务 id 在 ref_doc_id
    （Document.id_ 传下去的位置）。
    """
    return node.ref_doc_id or node.node_id


def build_index():
    """
    加载或重建实验索引（按语料指纹自动判断，幂等）。

    指纹变化（改了 CORPUS）时删除旧目录重建，避免向量与语料不一致。
    """
    fp = _corpus_fingerprint()
    fp_file = os.path.join(PERSIST_DIR, ".fingerprint")

    if os.path.exists(PERSIST_DIR) and os.path.exists(fp_file):
        with open(fp_file, "r", encoding="utf-8") as f:
            if f.read().strip() == fp:
                storage = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
                index = load_index_from_storage(storage)
                print(f"📂 语料未变，加载已有索引：{len(index.docstore.docs)} 个节点")
                return index
        print("🔄 语料已变更，重建索引...")
        shutil.rmtree(PERSIST_DIR)

    print(f"📚 构建索引：{len(CORPUS)} 条语料（首次需下载/加载 bge 模型）...")
    documents = [
        Document(text=text, metadata={"topic": topic}, id_=doc_id)
        for doc_id, topic, text in CORPUS
    ]
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    with open(fp_file, "w", encoding="utf-8") as f:
        f.write(fp)
    print(f"✅ 索引构建完成，持久化到 {PERSIST_DIR}")
    return index


def build_configs(index):
    """
    构造四种检索配置，返回 {配置名: retrieve_fn}。

    每个 retrieve_fn(query) -> List[NodeWithScore]，已按最终顺序排好。
    """
    reranker = SentenceTransformerRerank(model=RERANKER_MODEL, top_n=TOP_K)
    bm25 = BM25Retriever.from_defaults(
        index=index,
        similarity_top_k=CANDIDATE_K,
        token_pattern=BM25_TOKEN_PATTERN,
        skip_stemming=True,
    )

    def vector(top_k: int):
        r = index.as_retriever(similarity_top_k=top_k)
        return lambda q: r.retrieve(q)

    def fusion(final_k: int):
        """
        向量 + BM25 双路召回，RRF 融合。

        mode 必须用 reciprocal_rerank：向量分数是 0~1 余弦相似度，
        BM25 分数是 0~5+ 的 TF-IDF 权重，量纲不同，simple 直接相加会让
        BM25 完全主导。RRF 只用排名融合，天然规避量纲问题。
        num_queries=1 关闭 LLM 查询扩展，保持纯检索。
        """
        r = QueryFusionRetriever(
            [index.as_retriever(similarity_top_k=CANDIDATE_K), bm25],
            num_queries=1,
            mode="reciprocal_rerank",
            similarity_top_k=final_k,
        )
        return lambda q: r.retrieve(q)

    return {
        "A. baseline": vector(TOP_K),
        "B. +rerank": lambda q: reranker.postprocess_nodes(vector(CANDIDATE_K)(q), query_str=q),
        "C. +hybrid": fusion(TOP_K),
        "D. hybrid+rerank": lambda q: reranker.postprocess_nodes(fusion(CANDIDATE_K)(q), query_str=q),
    }


def validate_eval_set(eval_set, source: str) -> list:
    """
    评测集自校验闸门：拦截会静默污染指标的三类坏数据，返回清洗后的评测集。

    规则全部源自实际事故（README 踩坑 15）：
      - expected 引用不存在的 doc_id -> 该条指标必然全错（RAGAS 错标事故，
        30/30 静默生效，就是因为加载时没有这道闸）
      - expected 数量 > TOP_K -> Recall@k 被结构性封顶，指标失真
      - 空 query -> 解析性问题
    内置评测集同样过闸（防手改笔误），外部 evalset.json 更不可信，必须严检。
    """
    valid_ids = {doc_id for doc_id, _, _ in CORPUS}
    cleaned, bad = [], []
    for query, expected, qtype in eval_set:
        problems = []
        if not query or not str(query).strip():
            problems.append("query 为空")
        unknown = [e for e in expected if e not in valid_ids]
        if unknown:
            problems.append(f"未知 doc_id: {unknown}")
        if len(expected) > TOP_K:
            problems.append(f"expected {len(expected)} 条 > TOP_K={TOP_K}，Recall 被结构性封顶")
        if problems:
            bad.append((query, problems))
            continue
        cleaned.append((str(query), list(expected), qtype))
    if bad:
        print(f"🚨 {source} 有 {len(bad)} 条坏数据被剔除：")
        for q, why in bad:
            print(f"   - {q!r}: {'; '.join(why)}")
    return cleaned


def load_eval_set(path: Path | str = EVALSET_PATH):
    """
    加载评测集：优先读 gen_evalset.py 产出的 evalset.json，否则回退内置 EVAL_SET。

    两条路径都过 validate_eval_set 闸门。外部文件的 ground truth 历史上出过
    系统性错标（RAGAS 30/30），所以外部数据不是免检，而是更要严检——
    宁可启动时红字报错，也不能让错标签静默产出假指标。

    evalset.json 格式：[{"query": "...", "expected": [...], "type": "..."}]，
    type 缺失时回退读 synthesizer 字段，再缺失标 "unlabeled"。
    """
    path = Path(path)
    if not path.exists():
        eval_set, source = EVAL_SET, "内置 EVAL_SET"
    else:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            eval_set = [
                (
                    item["query"],
                    item["expected"],
                    item.get("type") or item.get("synthesizer") or "unlabeled",
                )
                for item in data
            ]
            source = str(path)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            print(f"⚠️  {path} 解析失败（{exc}），回退内置 EVAL_SET")
            eval_set, source = EVAL_SET, "内置 EVAL_SET（回退）"

    cleaned = validate_eval_set(eval_set, source)
    if not cleaned:
        raise SystemExit(f"🚨 评测集 {source} 清洗后为空，中止评测（别拿坏数据出指标）")
    n_neg = sum(1 for _, e, _ in cleaned if not e)
    dist = {}
    for _, _, t in cleaned:
        if t != "negative":
            dist[t] = dist.get(t, 0) + 1
    dist_str = " ".join(f"{t}:{n}" for t, n in sorted(dist.items()))
    print(f"📥 评测集来源 {source}：{len(cleaned)} 条（正 {len(cleaned) - n_neg} / 负 {n_neg}）[{dist_str}]")
    return cleaned


def evaluate(name: str, retrieve_fn, eval_set) -> dict:
    """
    对评测集跑一个配置，返回总体指标 + 分类型指标 + 逐条明细。

    负样本（expected 为空）不计入指标分母，单独统计命中情况。
    分类型统计的价值：每类查询压测一种机制（见 EVAL_SET 注释），总体指标
    打平时，分类型仍能看出 A/B/C/D 各自在哪类查询上占优、差在哪。
    """
    def _avg(xs):
        return sum(xs) / len(xs) if xs else None

    buckets = {}  # type -> [hits, recalls, mrrs]
    details, neg_hits = [], []

    for query, expected, qtype in eval_set:
        nodes = retrieve_fn(query)[:TOP_K]
        got = [_doc_id(n.node) for n in nodes]

        if not expected:
            neg_hits.append(got)
            continue

        matched = set(got) & set(expected)
        hit = 1.0 if matched else 0.0
        recall = len(matched) / len(expected)

        rr = 0.0
        for rank, doc_id in enumerate(got, 1):
            if doc_id in expected:
                rr = 1.0 / rank
                break

        buckets.setdefault(qtype, [[], [], []])
        buckets[qtype][0].append(hit)
        buckets[qtype][1].append(recall)
        buckets[qtype][2].append(rr)
        details.append((query, expected, got, bool(matched)))

    all_hits = [h for b in buckets.values() for h in b[0]]
    all_recs = [r for b in buckets.values() for r in b[1]]
    all_mrrs = [m for b in buckets.values() for m in b[2]]

    return {
        "name": name,
        "hit@k": _avg(all_hits),
        "recall@k": _avg(all_recs),
        "mrr@k": _avg(all_mrrs),
        "by_type": {t: (_avg(h), _avg(r), _avg(m)) for t, (h, r, m) in buckets.items()},
        "details": details,
        "neg_hits": neg_hits,
    }


def print_details(results, eval_set):
    """打印逐条命中明细，直观对比各配置把哪个文档排到了前面。"""
    print("\n" + "=" * 78)
    print("📋 逐条命中明细（列表为 Top-3 命中顺序，✅/❌ 看是否命中期望）")
    print("=" * 78)

    pos = [(q, e, t) for q, e, t in eval_set if e]
    for qi, (query, expected, qtype) in enumerate(pos):
        print(f"\n🔍 [{qtype}] {query}")
        print(f"   期望命中: {expected}")
        for res in results:
            _, _, got, hit = res["details"][qi]
            print(f"   {'✅' if hit else '❌'} {res['name']:<18} {got}")


def run_threshold_sweep(index, eval_set):
    """
    阈值敏感性实验：把 Demo 2 的 SCORE_GAP_RATIO 启发式放到整套评测集上标定。

    两个维度分开测（作用对象都是向量余弦相似度；reranker 输出是 logits，
    量纲不同，比例阈值对它不成立，故只作用于纯向量检索）：
      1. 相对比例：保留 score >= ratio * top1。过滤只减不增，且 top1 定义了
         ratio 必然保留自己，所以 Hit@3 只在"top1 非期望、期望文档排 2~3 位
         被过滤"时才掉——真正敏感的指标是 Recall 和负样本误命中。
      2. 绝对门槛：top1 < floor 时整题拒答。相对比例对负样本无效（负样本的
         分数同样抱团、比例接近 1），只有绝对门槛才可能拒掉库外问题，
         代价是可能误拒正样本。

    检索结果只算一次并缓存，逐阈值复用，扫描本身近乎零成本。
    """
    retriever = index.as_retriever(similarity_top_k=TOP_K)
    cached = []
    print("⏳ 阈值实验：构建检索缓存（每个查询只检索一次）...")
    for query, expected, qtype in eval_set:
        nodes = retriever.retrieve(query)
        cached.append(
            (query, expected, qtype, [(_doc_id(n.node), n.score or 0.0) for n in nodes])
        )

    def _metrics(got, expected):
        matched = set(got) & set(expected)
        hit = 1.0 if matched else 0.0
        rr = 0.0
        for rank, d in enumerate(got, 1):
            if d in expected:
                rr = 1.0 / rank
                break
        return hit, len(matched) / len(expected), rr

    # ---- 维度 1：相对比例（Demo 2 的 SCORE_GAP_RATIO） ----
    print("\n" + "=" * 78)
    print("📐 相对阈值扫描：保留 score >= ratio * top1（ratio=0.00 即未过滤的 baseline A）")
    print("=" * 78)
    print(f"{'ratio':>6}{'Hit@3':>9}{'Recall@3':>11}{'MRR@3':>9}{'multi-doc':>11}{'负误命中':>10}")
    print("-" * 78)
    for ratio in RELATIVE_RATIOS:
        hits, recs, mrrs, multi_recs = [], [], [], []
        neg_pairs = 0
        for query, expected, qtype, results in cached:
            top1 = results[0][1] if results else 0.0
            got = [d for d, s in results if s >= ratio * top1]
            if not expected:
                neg_pairs += len(got)
                continue
            h, r, m = _metrics(got, expected)
            hits.append(h)
            recs.append(r)
            mrrs.append(m)
            if qtype == "multi-doc":
                multi_recs.append(r)
        n = len(hits)
        multi = sum(multi_recs) / len(multi_recs) if multi_recs else 0.0
        print(f"{ratio:>6.2f}{sum(hits)/n:>9.1%}{sum(recs)/n:>11.1%}"
              f"{sum(mrrs)/n:>9.3f}{multi:>11.1%}{neg_pairs:>8}/9")

    # ---- 维度 2：绝对门槛（top1 < floor 整题拒答） ----
    print("\n" + "=" * 78)
    print("📐 绝对门槛扫描：top1 低于 floor 时整题拒答（考察能否挡住库外问题）")
    print("=" * 78)
    print(f"{'floor':>6}{'Hit@3':>9}{'Recall@3':>11}{'MRR@3':>9}{'误拒正样本':>12}{'负拒答':>9}{'负误命中':>10}")
    print("-" * 78)
    for floor in ABSOLUTE_FLOORS:
        hits, recs, mrrs = [], [], []
        false_reject, neg_rejected, neg_pairs = 0, 0, 0
        for query, expected, qtype, results in cached:
            top1 = results[0][1] if results else 0.0
            got = [d for d, _ in results] if top1 >= floor else []
            if not expected:
                neg_rejected += 1 if not got else 0
                neg_pairs += len(got)
                continue
            false_reject += 1 if not got else 0
            h, r, m = _metrics(got, expected)
            hits.append(h)
            recs.append(r)
            mrrs.append(m)
        n = len(hits)
        print(f"{floor:>6.2f}{sum(hits)/n:>9.1%}{sum(recs)/n:>11.1%}{sum(mrrs)/n:>9.3f}"
              f"{false_reject:>9}/{n}{neg_rejected:>7}/3{neg_pairs:>8}/9")

    # ---- 分离度分析：绝对门槛是否存在完美分离点 ----
    pos_top1 = [rs[0][1] for _, e, _, rs in cached if e and rs]
    neg_top1 = [rs[0][1] for _, e, _, rs in cached if not e and rs]
    print("\nℹ️  top1 分数分布：正样本 [{:.3f}, {:.3f}]，负样本 [{:.3f}, {:.3f}]".format(
        min(pos_top1), max(pos_top1), min(neg_top1), max(neg_top1)))
    if min(pos_top1) > max(neg_top1):
        print("   ✅ 存在完美分离门槛：floor ∈ ({:.3f}, {:.3f}) 可零误拒、全拒负样本".format(
            max(neg_top1), min(pos_top1)))
    else:
        print("   ❌ 正负样本 top1 区间重叠，不存在零代价的分离门槛，只能权衡误拒/误命中")


def main(sweep: bool = False):
    print("=" * 78)
    print("🧪 RAG 检索优化实验 - reranker + BM25 混合检索（纯检索评估）")
    print("=" * 78)

    # embed_model 在 main 才加载：被 gen_evalset.py import 时不必付出模型加载成本
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")
    eval_set = load_eval_set()
    index = build_index()

    print(f"⚙️  加载 reranker（{RERANKER_MODEL}，首次需下载约 278MB）...")
    configs = build_configs(index)
    print("✅ 配置就绪\n")

    results = []
    for name, fn in configs.items():
        print(f"⏳ 评估 {name} ...")
        results.append(evaluate(name, fn, eval_set))

    n_pos = len([e for _, e, _ in eval_set if e])
    print("\n" + "=" * 78)
    print(f"📊 检索质量对比（语料 {len(CORPUS)} 条，K={TOP_K}，候选池={CANDIDATE_K}，正样本 {n_pos} 条）")
    print("=" * 78)
    print(f"{'配置':<20}{'Hit@3':>10}{'Recall@3':>12}{'MRR@3':>10}")
    print("-" * 78)
    baseline = results[0]
    for res in results:
        print(f"{res['name']:<20}{res['hit@k']:>10.1%}{res['recall@k']:>12.1%}{res['mrr@k']:>10.3f}")
    print("-" * 78)
    for res in results[1:]:
        d_hit = res["hit@k"] - baseline["hit@k"]
        d_rec = res["recall@k"] - baseline["recall@k"]
        d_mrr = res["mrr@k"] - baseline["mrr@k"]
        print(f"   {res['name']} vs baseline: Hit {d_hit:+.1%}, Recall {d_rec:+.1%}, MRR {d_mrr:+.3f}")

    # 分类型报表：总体打平时，看各类查询（= 各压测机制）上谁占优
    type_names = sorted({t for res in results for t in res["by_type"]})
    print("\n📊 分类型 Hit@3 / Recall@3（类别与压测机制的对应见 EVAL_SET 注释）")
    print("-" * 78)
    print(f"{'type':<14}" + "".join(f"{res['name']:>24}" for res in results))
    for t in type_names:
        row = f"{t:<14}"
        for res in results:
            m = res["by_type"].get(t)
            row += f"{m[0]:>12.0%}/{m[1]:<11.0%}" if m else f"{'--':>24}"
        print(row)
    print("-" * 78)

    print_details(results, eval_set)

    neg_queries = [q for q, e, _ in eval_set if not e]
    if neg_queries:
        print("\n" + "=" * 78)
        print(f"🚫 负样本表现（{len(neg_queries)} 条无答案查询，看各配置误命中什么）")
        print("=" * 78)
        for qi, q in enumerate(neg_queries):
            print(f"\n🔍 {q}")
            for res in results:
                print(f"   {res['name']:<18} {res['neg_hits'][qi]}")

    if sweep:
        run_threshold_sweep(index, eval_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 检索优化实验")
    parser.add_argument(
        "--sweep", action="store_true",
        help="四配置对比后附加运行阈值敏感性实验（标定 Demo2 的 SCORE_GAP_RATIO）",
    )
    main(sweep=parser.parse_args().sweep)
