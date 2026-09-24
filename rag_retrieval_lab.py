#!/usr/bin/env python3
"""
RAG 检索优化实验 - reranker + BM25 混合检索

纯检索评估脚本：不加载 LLM、不生成答案，只对比四种检索配置的排序质量。

四种配置：
  A. baseline      向量检索 Top-3（现有 demo 方案，对照组）
  B. +rerank       向量检索 Top-8 -> bge-reranker 重排 -> Top-3
  C. +hybrid       向量 Top-8 + BM25 Top-8 -> RRF 融合 -> Top-3
  D. hybrid+rerank 向量 + BM25 -> RRF 融合 Top-8 -> reranker 重排 -> Top-3

指标（对每个查询算，再取宏平均）：
  Hit@3     前 3 是否至少命中一个相关文档
  Recall@3  前 3 覆盖了多少比例的相关文档
  MRR@3     第一个命中的倒数排名（衡量"命中得早不早"）

重要前提：知识库只有 8 条，检索候选池≈全集，因此本实验测的是
**排序能力**而非召回能力。Recall@3 的理论上限受"3 个坑位装 8 条"限制。
要观察召回率差异，需先把知识库扩充到数百条以上。

运行：
  set HF_ENDPOINT=https://hf-mirror.com
  uv run python rag_retrieval_lab.py
首次运行会下载 bge-reranker-base（约 278MB）。
"""

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever

# 纯检索实验，显式禁用 LLM：QueryFusionRetriever 在 num_queries=1 时
# 不会调用 LLM，但仍会 resolve 一个 Settings.llm 占位（MockLLM，零成本）
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")
Settings.llm = None

PERSIST_DIR = os.getenv("LLAMAINDEX_PATH", "./data/llamaindex_store")

# BM25 中文分词方案（实测结论，见 README 踩坑经验）：
# 默认 token_pattern 是 (?u)\b\w\w+\b，中文无空格会被当成整句单 token，
# 导致中文查询 BM25 得分全为 0。这里改成"中文按单字 + 英文按词"切分，
# 并跳过英文词干化（中文场景下 stemming 无意义且可能干扰）。
BM25_TOKEN_PATTERN = r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+"

# reranker：cross-encoder，输出的是 logits（约 -10~+10），不是概率。
# 仅用于排序；展示时经 sigmoid 归一化。
RERANKER_MODEL = "BAAI/bge-reranker-base"

# 候选池大小：知识库共 8 条，取 8 即"全部候选交给重排/融合"
CANDIDATE_K = 8
# 最终评估的坑位数
TOP_K = 3

# ============================================================
# 评测集：query -> 应命中的 doc_id（与 KNOWLEDGE_BASE 的 id 对应）
# 覆盖三类：单文档命中 / 多文档命中 / 无关查询（负样本）
# ============================================================

EVAL_SET = [
    ("LangGraph和CrewAI有什么区别", ["doc1", "doc2"]),
    ("如何给Agent添加长期记忆", ["doc1"]),
    ("企业级Agent架构怎么设计", ["doc5"]),
    ("本地运行的轻量模型有哪些", ["doc8"]),
    ("ReAct和Plan-and-Execute哪个更灵活", ["doc6", "doc7"]),
    ("向量数据库有哪些主流选择", ["doc3"]),
    ("RAG能解决什么问题", ["doc4"]),
    ("CrewAI支持哪几种流程", ["doc2"]),
    ("Chroma和Milvus各有什么特点", ["doc3"]),
    ("State、Node、Edge是什么概念", ["doc1"]),
    ("今天天气怎么样", []),  # 负样本：知识库无相关内容
]


def _doc_id(node) -> str:
    """
    取节点对应的文档标识（doc1..doc8）。

    注意：node.id_ 是构建时生成的 UUID，不是我们写的 doc1；
    真正的业务 id 在 node.ref_doc_id（Document.id_ 传下去的位置）。
    """
    return node.ref_doc_id or node.node_id


def load_index():
    """加载已持久化的向量索引（复用 demo 建好的，不重建）。"""
    if not os.path.exists(PERSIST_DIR):
        raise SystemExit(
            f"❌ 未找到索引目录 {PERSIST_DIR}，请先运行 rag_demo_llamaindex.py 构建索引"
        )
    storage = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage)
    print(f"📂 已加载索引：{len(index.docstore.docs)} 个节点")
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

    def vector_retriever(top_k: int):
        r = index.as_retriever(similarity_top_k=top_k)
        return lambda q: r.retrieve(q)

    def fusion_retriever(final_k: int):
        """
        向量 + BM25 双路召回，RRF 融合。

        mode 必须用 RECIPROCAL_RERANK：向量分数是 0~1 余弦相似度，
        BM25 分数是 0~5+ 的 TF-IDF 权重，量纲不同，SIMPLE 直接相加会让
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
        "A. baseline": vector_retriever(TOP_K),
        "B. +rerank": lambda q: reranker.postprocess_nodes(
            vector_retriever(CANDIDATE_K)(q), query_str=q
        ),
        "C. +hybrid": fusion_retriever(TOP_K),
        "D. hybrid+rerank": lambda q: reranker.postprocess_nodes(
            fusion_retriever(CANDIDATE_K)(q), query_str=q
        ),
    }


def sigmoid(x: float) -> float:
    """把 reranker 的 logits 压到 0~1，仅用于展示。"""
    import math

    return 1 / (1 + math.exp(-x))


def evaluate(name: str, retrieve_fn) -> dict:
    """
    对评测集跑一个配置，返回宏平均指标与逐条明细。

    负样本（expected 为空）不计入三个指标的分母，单独统计其命中情况。
    """
    hits, recalls, mrrs = [], [], []
    details = []
    neg_hits = []

    for query, expected in EVAL_SET:
        nodes = retrieve_fn(query)[:TOP_K]
        got = [_doc_id(n.node) for n in nodes]

        if not expected:
            neg_hits.append(got)
            continue

        matched = set(got) & set(expected)
        hits.append(1.0 if matched else 0.0)
        recalls.append(len(matched) / len(expected))

        rr = 0.0
        for rank, doc_id in enumerate(got, 1):
            if doc_id in expected:
                rr = 1.0 / rank
                break
        mrrs.append(rr)

        details.append((query, expected, got, bool(matched)))

    n = len(hits)
    return {
        "name": name,
        "hit@k": sum(hits) / n,
        "recall@k": sum(recalls) / n,
        "mrr@k": sum(mrrs) / n,
        "details": details,
        "neg_hits": neg_hits,
    }


def print_case_details(results):
    """打印逐条命中明细，直观对比各配置把哪个文档排到了前面。"""
    print("\n" + "=" * 78)
    print("📋 逐条命中明细（→ 表示排序后的命中顺序）")
    print("=" * 78)

    queries = [q for q, e in EVAL_SET if e]
    for qi, query in enumerate(queries):
        expected = [e for q, e in EVAL_SET if q == query][0]
        print(f"\n🔍 {query}")
        print(f"   期望命中: {expected}")
        for res in results:
            got = res["details"][qi][2]
            mark = "✅" if res["details"][qi][3] else "❌"
            print(f"   {mark} {res['name']:<18} {got}")


def main():
    print("=" * 78)
    print("🧪 RAG 检索优化实验 - reranker + BM25 混合检索（纯检索评估）")
    print("=" * 78)

    index = load_index()

    print(f"⚙️  正在加载 reranker（{RERANKER_MODEL}，首次需下载约 278MB）...")
    configs = build_configs(index)
    print("✅ 配置就绪\n")

    results = []
    for name, fn in configs.items():
        print(f"⏳ 评估 {name} ...")
        results.append(evaluate(name, fn))

    # 指标对比表
    print("\n" + "=" * 78)
    print(f"📊 检索质量对比（K={TOP_K}，候选池={CANDIDATE_K}，正样本 {len([e for _, e in EVAL_SET if e])} 条）")
    print("=" * 78)
    print(f"{'配置':<20}{'Hit@3':>10}{'Recall@3':>12}{'MRR@3':>10}")
    print("-" * 78)
    baseline = results[0]
    for res in results:
        print(
            f"{res['name']:<20}"
            f"{res['hit@k']:>10.1%}"
            f"{res['recall@k']:>12.1%}"
            f"{res['mrr@k']:>10.3f}"
        )
    print("-" * 78)
    for res in results[1:]:
        d_hit = res["hit@k"] - baseline["hit@k"]
        d_mrr = res["mrr@k"] - baseline["mrr@k"]
        print(
            f"   {res['name']} vs baseline: Hit@3 {d_hit:+.1%}, MRR@3 {d_mrr:+.3f}"
        )

    print_case_details(results)

    # 负样本表现：知识库无相关内容时，各配置会误命中什么
    print("\n" + "=" * 78)
    print("🚫 负样本表现（'今天天气怎么样' 知识库无答案，看各配置误命中什么）")
    print("=" * 78)
    for res in results:
        print(f"   {res['name']:<18} {res['neg_hits'][0]}")

    print("\n💡 结论提示：")
    print("   - 本实验知识库仅 8 条，候选池≈全集，测的是【排序能力】而非召回能力")
    print("   - 要观察召回率差异，需先把知识库扩充到数百条以上")


if __name__ == "__main__":
    main()
