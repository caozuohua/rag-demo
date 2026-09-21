#!/usr/bin/env python3
"""
RAG Demo - LlamaIndex 版本（本地 LLM RAG 完整版）

用 LlamaIndex 原生组件实现一个真正端到端的 RAG：
1. Document + VectorStoreIndex 完成文档向量化与本地持久化
2. Retriever 做语义检索，返回 Top-K 最相似节点
3. 把命中文档拼成上下文，交给本地 LLM (llama.cpp + Qwen2.5) 生成答案

组件对应关系（Chroma 版 -> LlamaIndex 版）：
  Collection.add()           -> index.insert() / 构建索引
  Collection.query()         -> retriever.retrieve()
  PersistentClient 落盘      -> storage_context.persist() 落盘
  DefaultEmbeddingFunction   -> HuggingFaceEmbedding（bge-small-zh 本地推理）
  (无 LLM)                   -> LlamaCPP（llama.cpp + Qwen2.5 GGUF）

依赖安装：
  uv add llama-index-core llama-index-embeddings-huggingface llama-index-llms-llama-cpp
  # llama-cpp-python 用预编译 wheel，避免本地编译：
  uv pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
运行：
  uv run python rag_demo_llamaindex.py
首次运行会下载 bge-small-zh 中文向量模型（约 100MB）；
国内网络可先设置：set HF_ENDPOINT=https://hf-mirror.com
"""

import os

from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP

# 本地中文向量模型：句子级语义 Embedding，CPU 可跑，无需 API Key
# 与原版的 all-MiniLM 相比，bge-small-zh 对中文知识库检索效果更好
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")

# 本地 LLM：llama.cpp 加载 Qwen2.5-1.5B-Instruct GGUF
# MODEL_PATH 走环境变量，默认指向 download_model.py 下载的模型
MODEL_PATH = os.getenv("MODEL_PATH", "./models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
Settings.llm = LlamaCPP(
    model_path=MODEL_PATH,
    temperature=0.3,        # 降到 0.3 抑制小模型的发散与循环
    max_new_tokens=512,     # 答案长度上限，避免无限生成
    context_window=4096,    # 4K 上下文足够装下检索片段+问题+答案
    model_kwargs={"n_gpu_layers": 0},  # 纯 CPU；有 GPU 可调大让 llama.cpp 自动 offload
    generate_kwargs={"repeat_penalty": 1.1},  # 抑制循环式复读
    verbose=False,
)


def _qwen_chat_prompt(system: str, user: str) -> str:
    """
    手工拼 Qwen2.5 的 ChatML 模板。

    直接用 llm.complete() 喂裸 prompt 给 instruct 模型会出问题：
      - 复读 prompt 里的"问题:回答:"指令字面
      - 在小模型上陷入循环式重复
    显式套 ChatML 后，模型按对话角色理解指令，输出更稳定。
    """
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

# 索引持久化目录，走环境变量，方便 clone 即跑
PERSIST_DIR = os.getenv("LLAMAINDEX_PATH", "./data/llamaindex_store")

# ============================================================
# 知识库内容（模拟企业文档）
# Document 三要素：
#   id_      - 文档唯一标识，幂等写入时据此去重
#   text     - 原文，构建索引时被 Embedding 转向量
#   metadata - 附属标签，后续可按字段过滤检索
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


def _build_documents():
    """把字典形式的知识库转成 LlamaIndex Document 列表。"""
    return [
        Document(text=doc["text"], metadata=doc["metadata"], id_=doc["id"])
        for doc in KNOWLEDGE_BASE
    ]


def init_index():
    """
    加载已有索引；不存在则用知识库构建并落盘（幂等）。

    - load_index_from_storage：从 persist 目录恢复索引，无需重新 Embedding
    - VectorStoreIndex：构建时自动完成 切分 -> 向量化 -> 建索引
    - persist：把向量、文档、元数据写到磁盘，下次直接加载
    """
    if os.path.exists(PERSIST_DIR):
        print(f"📂 发现已有索引，从 {PERSIST_DIR} 加载")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        return load_index_from_storage(storage_context)

    print("📚 未发现索引，正在构建（首次需要下载模型并向量化）...")
    documents = _build_documents()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print(f"✅ 索引构建完成，已持久化到 {PERSIST_DIR}")
    return index


def search_knowledge(index, query: str, top_k: int = 3):
    """
    语义检索：query 向量化后取最相似的 top_k 个节点。

    Args:
        index: 已加载/构建好的 VectorStoreIndex
        query: 用户问题文本
        top_k: 返回节点数量

    Returns:
        NodeWithScore 列表，每个元素包含：
          node.node.get_content() - 命中文本
          node.metadata          - 元数据
          node.score             - 余弦相似度，越大越相似（与 Chroma 的距离相反）
    """
    retriever = index.as_retriever(similarity_top_k=top_k)
    return retriever.retrieve(query)


def simple_rag(index, question: str):
    """
    完整 RAG 演示：检索 -> 拼上下文 -> 本地 LLM 生成。

    与 Chroma 版的差异：LlamaIndex 的 score 直接是相似度（越大越好），
    不需要再做 1-distance 换算。
    """
    print(f"\n🔍 问题：{question}")
    print("-" * 50)

    nodes = search_knowledge(index, question, top_k=3)

    print("📖 检索到的相关知识：\n")
    docs = []
    for i, node_with_score in enumerate(nodes, 1):
        node = node_with_score.node
        score = node_with_score.score or 0.0
        topic = node.metadata.get("topic", "未分类")
        text = node.get_content()
        docs.append(text)

        print(f"  [{i}] 相似度: {score:.2%} | 主题: {topic}")
        print(f"      {text[:80]}...")
        print()

    # 把命中文档拼成 LLM 的上下文，约束 LLM 只用检索内容作答
    context = "\n".join([f"- {doc}" for doc in docs])
    system_msg = (
        "你是一个知识助手，只依据下面给出的检索知识回答问题，"
        "不要编造检索以外的内容。如果检索知识与问题无关，"
        "直接说「知识库中没有相关信息」。"
    )
    user_msg = (
        f"检索到的知识：\n{context}\n\n"
        f"问题：{question}\n\n"
        "请用中文简洁回答。"
    )
    prompt = _qwen_chat_prompt(system_msg, user_msg)

    print("💡 本地 LLM 生成的回答：")
    print("-" * 50)
    response = Settings.llm.complete(prompt)
    print(response.text)


def main():
    print("=" * 60)
    print("🤖 RAG Demo - LlamaIndex + bge-small-zh + Qwen2.5 (llama.cpp)")
    print("=" * 60)

    # 步骤1: 加载或构建索引（幂等，已有索引直接读盘不重复向量化）
    index = init_index()

    # 步骤2: 对一批示例问题做完整 RAG：检索 + LLM 生成
    questions = [
        "LangGraph和CrewAI有什么区别？",
        "如何给Agent添加长期记忆？",
        "企业级Agent架构怎么设计？",
        "本地运行的轻量模型有哪些？",
    ]

    for q in questions:
        simple_rag(index, q)
        print("=" * 60)


if __name__ == "__main__":
    main()
