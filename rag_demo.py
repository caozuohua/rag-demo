#!/usr/bin/env python3
"""
RAG Demo - 向量数据库 + Agent
使用 Chroma + LangGraph + DeepSeek

功能：
1. 把知识文档存入Chroma向量库
2. Agent接收问题，检索相关知识
3. 结合检索结果生成回答
"""

import os
import chromadb
from chromadb.utils import embedding_functions

# ============================================================
# 知识库内容（模拟企业文档）
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
# 初始化Chroma（本地持久化）
# ============================================================

def init_vectordb():
    """初始化向量数据库"""
    # 使用本地持久化存储，路径优先读环境变量
    db_path = os.getenv("CHROMA_PATH", "./data/chroma_db")
    client = chromadb.PersistentClient(path=db_path)

    # 使用默认Embedding（sentence-transformers，无需API Key）
    ef = embedding_functions.DefaultEmbeddingFunction()

    # 创建或获取集合
    collection = client.get_or_create_collection(
        name="agent_knowledge",
        embedding_function=ef,
        metadata={"description": "AI Agent开发知识库"}
    )

    return collection


def load_knowledge(collection):
    """加载知识到向量库"""
    # 检查是否已有数据
    count = collection.count()
    if count > 0:
        print(f"✅ 知识库已有 {count} 条记录，跳过加载")
        return

    print("📚 正在加载知识库...")
    collection.add(
        ids=[doc["id"] for doc in KNOWLEDGE_BASE],
        documents=[doc["text"] for doc in KNOWLEDGE_BASE],
        metadatas=[doc["metadata"] for doc in KNOWLEDGE_BASE]
    )
    print(f"✅ 已加载 {len(KNOWLEDGE_BASE)} 条知识")


def search_knowledge(collection, query: str, n_results: int = 3):
    """语义检索"""
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results


# ============================================================
# 简单RAG（不需要API Key的版本）
# ============================================================

def simple_rag(collection, question: str):
    """简单RAG：检索 + 展示（无需LLM API）"""
    print(f"\n🔍 问题：{question}")
    print("-" * 50)

    # 检索相关知识
    results = search_knowledge(collection, question, n_results=3)

    docs = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    print("📖 检索到的相关知识：\n")
    for i, (doc, dist, meta) in enumerate(zip(docs, distances, metadatas), 1):
        similarity = 1 - dist  # 转换为相似度
        print(f"  [{i}] 相似度: {similarity:.2%} | 主题: {meta['topic']}")
        print(f"      {doc[:80]}...")
        print()

    # 构建上下文
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

    # 初始化
    collection = init_vectordb()
    load_knowledge(collection)

    print(f"\n📊 知识库统计：{collection.count()} 条记录\n")

    # 测试问题
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
