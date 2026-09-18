#!/usr/bin/env python3
"""
RAG Demo - 向量数据库 + 知识检索（无 LLM 版）

最小可运行的 RAG 检索环节演示：
1. 将知识文档向量化并存入 Chroma 持久化向量库
2. 对用户问题做语义检索，返回 Top-K 最相似文档
3. 展示检索结果与相似度（不调用 LLM，便于离线学习）

依赖：chromadb（默认 Embedding 用 ONNX MiniLM，首次运行会自动下载模型）
运行：uv run python rag_demo.py
"""

import os
import chromadb
from chromadb.utils import embedding_functions

# ============================================================
# 知识库内容（模拟企业文档）
# 每条记录三个字段：
#   id       - 文档唯一标识，向量库据此去重/更新
#   text     - 原文，会被 Embedding 转成向量
#   metadata - 附属标签，检索时可作过滤条件
# ============================================================

KNOWLEDGE_BASE = [
    {
        "id": "doc1",
        "text": "LangGraph是基于图结构的Agent编排框架，核心概念包括State（状态）、Node（节点）、Edge（边）。State是整个工作流的共享记忆，Node是执行步骤，Edge定义执行顺序。",
        "metadata": {"topic": "LangGraph", "type": "概念"}
    },
    {
        "id": "doc2",
        "text": "CrewAI是角色驱动的多Agent框架，每个Agent有role（角色）、goal（目标）、backstory（背景故事）。支持sequential（顺序）和hierarchical（层级）两种流程。",
        "metadata": {"topic": "CrewAI", "type": "概念"}
    },
    {
        "id": "doc3",
        "text": "向量数据库通过Embedding将文本转为高维向量，支持语义相似度检索。主流选择：Chroma（本地轻量）、Milvus（企业级）、Qdrant（高性能）、LanceDB（嵌入式）。",
        "metadata": {"topic": "向量数据库", "type": "概念"}
    },
    {
        "id": "doc4",
        "text": "RAG（检索增强生成）流程：文档切分→Embedding→存入向量库→用户提问→检索Top-K→LLM结合上下文生成回答。解决LLM知识截止和幻觉问题。",
        "metadata": {"topic": "RAG", "type": "技术"}
    },
    {
        "id": "doc5",
        "text": "企业级Agent架构分五层：应用层（智能客服/RPA）、编排层（多Agent协作）、Agent层（Planner/Executor）、基础设施层（LLM/工具/知识库）、企业集成层（SSO/权限/审计）。",
        "metadata": {"topic": "企业级Agent", "type": "架构"}
    },
    {
        "id": "doc6",
        "text": "ReAct算法：Reasoning + Acting循环。思考（Thought）→行动（Action）→观察（Observation）→再思考。适合探索性任务，灵活性高。",
        "metadata": {"topic": "ReAct", "type": "算法"}
    },
    {
        "id": "doc7",
        "text": "Plan-and-Execute算法：先规划（Planner生成步骤列表），再执行（Executor逐步执行）。适合长流程任务，效率高，但灵活性低于ReAct。",
        "metadata": {"topic": "Plan-and-Execute", "type": "算法"}
    },
    {
        "id": "doc8",
        "text": "Qwen2.5-0.5B是阿里云开源的轻量级模型，约1GB大小，支持本地运行，中文优化好。通过Ollama可以一键部署，适合本地写作辅助等场景。",
        "metadata": {"topic": "本地模型", "type": "工具"}
    },
]

# ============================================================
# 向量库初始化与知识加载
# ============================================================

def init_vectordb():
    """
    初始化 Chroma 向量库并返回指定集合。

    - PersistentClient：数据落盘到磁盘，进程退出不丢
    - 路径走环境变量 CHROMA_PATH（默认 ./data/chroma_db），
      避免硬编码绝对路径，方便 clone 即跑
    - DefaultEmbeddingFunction：ONNX MiniLM 本地推理，无需 API Key

    Returns:
        Collection 对象，类比 SQL 中的表，后续 add/query 都基于它
    """
    db_path = os.getenv("CHROMA_PATH", "./data/chroma_db")
    client = chromadb.PersistentClient(path=db_path)

    ef = embedding_functions.DefaultEmbeddingFunction()

    # get_or_create: 首次创建，后续复用，保证可重复运行不报错
    collection = client.get_or_create_collection(
        name="agent_knowledge",
        embedding_function=ef,
        metadata={"description": "AI Agent开发知识库"}
    )
    return collection


def load_knowledge(collection):
    """
    将 KNOWLEDGE_BASE 写入向量库；已存在数据则跳过（幂等）。

    add() 三列表一一对应：
      ids       - 文档定位/去重键
      documents - 原文，向量库内部会调用 Embedding 自动转向量
      metadatas - 元数据，检索时可作过滤条件
    """
    if collection.count() > 0:
        print(f"✅ 知识库已有 {collection.count()} 条记录，跳过加载")
        return

    print("📚 正在加载知识库...")
    collection.add(
        ids=[doc["id"] for doc in KNOWLEDGE_BASE],
        documents=[doc["text"] for doc in KNOWLEDGE_BASE],
        metadatas=[doc["metadata"] for doc in KNOWLEDGE_BASE]
    )
    print(f"✅ 已加载 {len(KNOWLEDGE_BASE)} 条知识")


def search_knowledge(collection, query: str, n_results: int = 3):
    """
    语义检索：把 query 向量化后，在向量库中找最相似的 n_results 条文档。

    Args:
        query: 用户问题文本
        n_results: 返回 Top-K 条结果

    Returns:
        Chroma query 结果字典，常用键：
          documents - 嵌套列表，外层按 query 顺序，内层是该 query 的命中文档
          distances - 对应的余弦距离（越小越相似）
          metadatas - 对应的元数据
    """
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results


# ============================================================
# 简单 RAG（无 LLM 版）
# ============================================================

def simple_rag(collection, question: str):
    """
    最小 RAG 演示：检索 → 展示。

    不调用 LLM，仅把检索到的文档作为"候选上下文"打印出来，
    便于直观感受向量检索的效果与相似度分布。
    真正的 RAG 还会把 context 拼进 prompt 交给 LLM 生成答案。
    """
    print(f"\n🔍 问题：{question}")
    print("-" * 50)

    results = search_knowledge(collection, question, n_results=3)

    # query 传的是单条，所以取 [0] 拿到该问题的命中列表
    docs = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    print("📖 检索到的相关知识：\n")
    for i, (doc, dist, meta) in enumerate(zip(docs, distances, metadatas), 1):
        # Chroma 返回余弦距离（0=完全相同，2=完全相反），1-dist 转成相似度百分比
        similarity = 1 - dist
        print(f"  [{i}] 相似度: {similarity:.2%} | 主题: {meta['topic']}")
        print(f"      {doc[:80]}...")
        print()

    # 把命中文档拼成 LLM 的上下文（此处只打印，未实际调用 LLM）
    context = "\n".join([f"- {doc}" for doc in docs])

    print("💡 基于以上知识，回答如下：")
    print("-" * 50)
    print(f"根据知识库检索，关于「{question}」的相关信息：\n")
    for doc in docs:
        print(f"• {doc}\n")


# ============================================================
# 主程序
# ============================================================

def main():
    print("=" * 60)
    print("🤖 RAG Demo - Chroma向量数据库 + 知识检索")
    print("=" * 60)

    # 步骤1: 建库 + 加载知识（幂等，重复运行不会重复写入）
    collection = init_vectordb()
    load_knowledge(collection)

    print(f"\n📊 知识库统计：{collection.count()} 条记录\n")

    # 步骤2: 对一批示例问题做检索，观察检索结果
    questions = [
        "LangGraph和CrewAI有什么区别？",
        "如何给Agent添加长期记忆？",
        "企业级Agent架构怎么设计？",
        "本地运行的轻量模型有哪些？",
    ]

    for q in questions:
        simple_rag(collection, q)
        print("=" * 60)


if __name__ == "__main__":
    main()
