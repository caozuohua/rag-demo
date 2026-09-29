#!/usr/bin/env python3
"""
评测集生成器 - RAGAS + GLM-4.5-Air（智谱 OpenAI 兼容接口）

一次性离线动作：读 rag_retrieval_lab 的 CORPUS -> RAGAS 自动生成问答 ->
映射回 doc_id -> 写 evalset.json，供 rag_retrieval_lab.py 自动加载。

设计权衡：
  - 出题 LLM 用云端强模型：知识图谱抽取和问题进化要求稳定的结构化输出，
    本地 1.5B 做不了（会产大量错误 ground truth 污染评测集）
  - Embedding 复用本地 bge-small-zh（RAGAS 用它做问题去重/聚类），零新增下载
  - 生成完即固化成 JSON：日常实验全本地，不再碰 API

doc_id 映射说明：
  RAGAS 输出的 reference_contexts 是原文片段（会被它的切块器重切），
  这里用"原文子串匹配"映射回 CORPUS 的 doc_id；映射失败的条目直接丢弃，
  保证 evalset.json 里每条的 ground truth 都可信。

前置：
  项目根建 .env 文件：ZHIPUAI_API_KEY=your-key
  （PowerShell 下 set 命令不生效，见 README 踩坑 12）
运行：
  uv run python gen_evalset.py            # 默认生成 30 条
  uv run python gen_evalset.py --size 50
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# 路径一律锚定项目根，不依赖运行时 CWD（从别的目录执行也不会写错位置）
PROJECT_ROOT = Path(__file__).resolve().parent
EVALSET_PATH = PROJECT_ROOT / "evalset.json"

load_dotenv(PROJECT_ROOT / ".env")  # 优先读项目根 .env；已有同名环境变量则不覆盖

from langchain_core.documents import Document  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.testset import TestsetGenerator  # noqa: E402

# 复用 lab 的语料定义（rag_retrieval_lab 已做轻量 import 设计，此处不触发模型加载）
from rag_retrieval_lab import CORPUS

# 智谱 OpenAI 兼容接口；GLM-4.5-Air 是 Air 版（3.5B 激活参数），性价比取向
# 模型 id 以智谱开放平台为准，可用环境变量 GLM_MODEL 覆盖
GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
GLM_MODEL = os.getenv("GLM_MODEL", "glm-4.5-air")

# 负样本：知识库中不存在答案的问题，检验系统"该拒答时是否误命中"。
# RAGAS 只造正样本，负样本继续手工维护（这类样本靠人对领域的理解）。
NEGATIVE_QUERIES = [
    "今天天气怎么样",
    "推荐一部最近上映的电影",
]


def build_langchain_docs(topics=None):
    """CORPUS -> 按主题合并的 langchain Document 列表。

    RAGAS 要求文档 >=100 token（default.py 的长度分档检查），而 CORPUS 是
    为检索评测优化的原子短句（实测 26-59 token/条），直接喂会触发
    "Documents too short" 报错。这里把同一主题的若干句子拼成一篇"主题文档"，
    跨过门槛。metadata.member_ids 记录该文档由哪些 CORPUS doc_id 组成，
    供映射时判断 context 粒度（整篇 vs 单句）。

    topics: 只保留指定主题（探针实验用），None 表示全部主题。
    """
    from collections import OrderedDict

    groups = OrderedDict()
    for doc_id, topic, text in CORPUS:
        groups.setdefault(topic, []).append((doc_id, text))

    docs = []
    for topic, members in groups.items():
        if topics and topic not in topics:
            continue
        joined = "\n".join(t for _, t in members)
        docs.append(
            Document(
                page_content=joined,
                metadata={
                    "topic": topic,
                    "filename": topic,
                    "member_ids": [d for d, _ in members],
                },
            )
        )
    return docs


def contexts_to_doc_ids(contexts):
    """
    把 RAGAS 返回的 reference_contexts（原文片段）映射回 CORPUS 的 doc_id。

    匹配规则：片段是某条语料的子串，或某条语料包含片段（切块可能截断）。
    返回去重后的 doc_id 集合；一条都匹配不上则返回空集合（该条会被丢弃）。

    粒度探针意义：文档已按主题合并，若 RAGAS 返回整篇合并文档作为 context，
    这里会命中该主题的多个 member_id（粗粒度）；若返回单句，只命中一个（细粒度）。
    命中的 id 数直接反映 context 粒度。
    """
    ids = set()
    for ctx in contexts:
        ctx = ctx.strip()
        if not ctx:
            continue
        for doc_id, _, text in CORPUS:
            if ctx in text or text in ctx:
                ids.add(doc_id)
                break
    return ids


def resolve_local_model(model_name: str):
    """
    在常见 HF 缓存目录里定位已下载的模型 snapshot，返回本地路径；找不到返回 None。

    背景：LlamaIndex 的 HuggingFaceEmbedding 会把模型下到它自己的缓存目录
    （%LOCALAPPDATA%/llama_index/.../Cache），而 langchain 的 HuggingFaceEmbeddings
    默认只查标准 HF 缓存（HF_HOME 或 ~/.cache/huggingface）。两者位置不一致时，
    离线模式（HF_HUB_OFFLINE=1）下 langchain 会因"标准位置没有"而直接报错。
    这里主动搜遍两个位置，命中就把本地 snapshot 路径喂给 embedding，绕开 hub 名称查找。
    """
    org, _, name = model_name.partition("/")
    dir_tag = f"models--{org}--{name}" if name else f"models--{org}"

    candidates = [
        os.getenv("HF_HOME"),
        os.path.expanduser("~/.cache/huggingface/hub"),
    ]
    # LlamaIndex 把模型下到自己的缓存目录；该位置只在 Windows 下存在，
    # 非 Windows 直接跳过，避免退化成相对路径去扫 CWD
    if sys.platform == "win32":
        candidates.append(
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "llama_index")
        )
    for root in filter(None, candidates):
        # snapshot 目录需同时含 config.json 与模型权重才算完整
        pattern = os.path.join(root, "**", dir_tag, "snapshots", "*")
        for snap in glob.glob(pattern, recursive=True):
            if os.path.isfile(os.path.join(snap, "config.json")) and (
                os.path.isfile(os.path.join(snap, "model.safetensors"))
                or os.path.isfile(os.path.join(snap, "pytorch_model.bin"))
            ):
                return snap
    return None


def generate(size: int, probe: bool = False):
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "❌ 未读到 ZHIPUAI_API_KEY。请在项目根建 .env 文件写入：\n"
            "   ZHIPUAI_API_KEY=your-key\n"
            "（注意 PowerShell 里 set 命令不设置环境变量，详见 README 踩坑 12）"
        )

    # 环境变量是非可信输入，直接 int() 会在非数字时报裸 ValueError，这里给明确提示
    try:
        max_tokens = int(os.getenv("GLM_MAX_TOKENS", "8192"))
    except ValueError:
        raise SystemExit(
            "❌ GLM_MAX_TOKENS 必须是整数，当前值无法解析为数字，"
            "请检查环境变量或项目根 .env"
        )

    print(f"🔌 出题模型：{GLM_MODEL} @ {GLM_BASE_URL}")
    # max_tokens 必须给足：GLM-4.5-Air 是推理模型，thinking token 会吃掉输出预算，
    # 默认上限下 JSON 答案被截断 -> finish_reason="length" -> ragas 判为未完成，
    # 抛 LLMDidNotFinishException（其白名单只认 stop/MAX_TOKENS/eos_token）。
    llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=GLM_MODEL,
            base_url=GLM_BASE_URL,
            api_key=api_key,
            temperature=0.3,  # 出题要稳定可复现，压低温度
            max_tokens=max_tokens,
        )
    )

    # 本地 embedding：复用 bge-small-zh 做问题去重/聚类，不走云端
    # 优先定位已缓存的 snapshot（LlamaIndex 与 langchain 缓存位置不同，见 resolve_local_model）
    embed_name = "BAAI/bge-small-zh-v1.5"
    local = resolve_local_model(embed_name)
    target = local or embed_name
    print(f"🧬 去重 Embedding：{'本地缓存 ' + target if local else embed_name + '（将联网下载）'}...")
    embed = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=target,
            encode_kwargs={"normalize_embeddings": True},
        )
    )

    if probe:
        # 探针：只取 2 个"多句主题"（成员数>=3），生成 2 题，打印原始 context 粒度
        multi = [t for t in {tp for _, tp, _ in CORPUS}
                 if sum(1 for _, tt, _ in CORPUS if tt == t) >= 3]
        topics = multi[:2]
        docs = build_langchain_docs(topics=topics)
        print(f"🔬 探针模式：仅用 {len(docs)} 篇多句主题 {topics}，生成 2 题看 context 粒度")
        size = 2
    else:
        docs = build_langchain_docs()
        print(f"📚 语料 {len(docs)} 条，开始生成评测集（约 {size} 题，"
              f"知识图谱构建 + 问题进化会产生数百次 LLM 调用，请耐心等待）...")

    generator = TestsetGenerator(llm=llm, embedding_model=embed)
    # raise_exceptions=False：单个 sample 生成失败时跳过而非整体崩溃
    # （推理模型偶发 finish_reason=length 时，保住其余已成功的样本）
    dataset = generator.generate_with_langchain_docs(
        docs, testset_size=size, raise_exceptions=False
    )
    records = dataset.to_pandas().to_dict("records")

    if probe:
        print("\n" + "=" * 70)
        print("🔬 探针结果：每条问题的 reference_contexts 粒度分析")
        print("=" * 70)
        for i, row in enumerate(records, 1):
            ctxs = row.get("reference_contexts") or []
            ids = contexts_to_doc_ids(ctxs)
            print(f"\n[{i}] 问题：{row['user_input']}")
            print(f"    synthesizer: {row.get('synthesizer_name','')}")
            print(f"    context 条数: {len(ctxs)}  映射命中 doc_id 数: {len(ids)}")
            for j, c in enumerate(ctxs, 1):
                preview = c.replace("\n", " ⏎ ")
                print(f"      ctx{j}: {preview[:120]}")
            print(f"    命中 doc_id: {sorted(ids)}")
        print("\n判读：命中 doc_id 数==1 → 细粒度（可直接用于检索评测）；"
              "命中数>1 且 context 含 ⏎ 换行 → 整篇粗粒度（ground truth 会误标同组句）")
        return

    # 转换：user_input/reference_contexts -> query/expected(doc_id 列表)
    eval_set, dropped = [], 0
    for row in records:
        expected = contexts_to_doc_ids(row.get("reference_contexts") or [])
        if not expected:
            dropped += 1
            continue
        eval_set.append(
            {
                "query": row["user_input"],
                "expected": sorted(expected),
                "synthesizer": row.get("synthesizer_name", ""),
            }
        )

    # 负样本混入（expected 为空，lab 会单独统计不误计入指标分母）
    for q in NEGATIVE_QUERIES:
        eval_set.append({"query": q, "expected": [], "synthesizer": "manual_negative"})

    with open(EVALSET_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, ensure_ascii=False, indent=2)

    print(f"✅ 生成完成：正样本 {len(eval_set) - len(NEGATIVE_QUERIES)} 条"
          f"（丢弃无法映射的 {dropped} 条）+ 负样本 {len(NEGATIVE_QUERIES)} 条")
    print(f"📄 已写入 {EVALSET_PATH}，rag_retrieval_lab.py 下次运行自动加载")


def main():
    parser = argparse.ArgumentParser(description="RAGAS 评测集生成器")
    parser.add_argument("--size", type=int, default=30, help="生成题目数量（默认 30）")
    parser.add_argument("--probe", action="store_true",
                        help="探针模式：只用 2 篇多句主题生成 2 题，打印 context 粒度后退出")
    args = parser.parse_args()
    generate(args.size, probe=args.probe)


if __name__ == "__main__":
    main()
