#!/usr/bin/env python3
"""旅游助手 Agent（ReAct，极简版）。

- MCP Client：连接独立运行的天气 server（默认 http://127.0.0.1:8000/mcp，
  可用环境变量 MCP_SERVER_URL 覆盖），动态发现工具，不硬编码工具名
- 本地 RAG：search_docs 直连 Chroma 检索攻略
- 记忆：data/memory/history.txt 追加，启动时加载最近轮次

用法：
    python scripts/agent.py   # 交互式对话，q/exit 退出
"""

import asyncio
import json
import os
from pathlib import Path

import chromadb
import httpx
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

LLM_URL = "https://api.deepseek.com/chat/completions"
LLM_KEY = os.getenv("deepseek_key", "")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
MCP_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/mcp")
MAX_ITERS = 6

MEMORY_FILE = ROOT / "data/memory/history.txt"
MEMORY_LINES = 30

EMBED_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/embeddings/"
    "multimodal-embedding/multimodal-embedding"
)
EMBED_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("api_key", "")

BASE_SYSTEM = """你是旅游助手，帮用户规划行程、回答攻略问题。请用中文回答，简洁有条理。
下面是你可用的工具。需要实时信息（如天气）时调用相应工具；需要本地攻略知识时调用 search_docs 并注明来源。"""

# 本地 RAG 工具（不走 MCP）
LOCAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "检索本地旅游攻略文档，返回相关片段",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词或问题"},
                    "top_k": {"type": "integer", "description": "返回片段数，默认 3"},
                },
                "required": ["query"],
            },
        },
    }
]


# ---------- LLM ----------

def llm(messages: list[dict], tools: list[dict]) -> dict:
    resp = httpx.post(
        LLM_URL,
        headers={"Authorization": f"Bearer {LLM_KEY}"},
        json={"model": LLM_MODEL, "messages": messages, "tools": tools, "tool_choice": "auto"},
        timeout=60,
    )
    resp.raise_for_status()
    msg = resp.json()["choices"][0]["message"]
    return {k: v for k, v in msg.items() if v not in (None, "")}


# ---------- RAG ----------

def embed(texts: list[str]) -> list[list[float]]:
    resp = httpx.post(
        EMBED_URL,
        headers={"Authorization": f"Bearer {EMBED_KEY}"},
        json={
            "model": "qwen3-vl-embedding",
            "input": {"contents": [{"text": t} for t in texts]},
            "parameters": {"dimension": 1024},
        },
        timeout=60,
    )
    resp.raise_for_status()
    return [e["embedding"] for e in resp.json()["output"]["embeddings"]]


def search_docs(query: str, top_k: int = 3) -> str:
    collection = chromadb.PersistentClient(path=str(ROOT / "data/chroma")).get_collection(
        "knowledge"
    )
    result = collection.query(
        query_embeddings=embed([query]),
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    parts = []
    for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        parts.append(f"【来源 {meta.get('source', 'unknown')}】(相关度 {1 - dist:.2f})\n{doc}")
    return "\n\n".join(parts) if parts else "未检索到相关内容"


# ---------- 记忆 ----------

def load_memory() -> list[dict]:
    if not MEMORY_FILE.exists():
        return []
    out = []
    for line in MEMORY_FILE.read_text(encoding="utf-8").splitlines()[-MEMORY_LINES:]:
        if line.startswith("用户: "):
            out.append({"role": "user", "content": line[3:]})
        elif line.startswith("助手: "):
            out.append({"role": "assistant", "content": line[3:]})
    return out


def save_memory(user_text: str, answer: str) -> None:
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with MEMORY_FILE.open("a", encoding="utf-8") as f:
        f.write(f"用户: {user_text}\n助手: {answer}\n")


# ---------- Agent ----------

async def react(messages: list[dict], tools: list[dict], session: ClientSession) -> str:
    """ReAct 循环：LLM → 若有 tool_calls 则执行并回填，直到输出最终回答。"""
    for _ in range(MAX_ITERS):
        msg = await asyncio.to_thread(llm, messages, tools)
        messages.append(msg)
        calls = msg.get("tool_calls")
        if not calls:
            return msg.get("content", "").strip()

        for call in calls:
            name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"] or "{}")
            print(f"  [工具] {name}{args}")
            if name == "search_docs":  # 本地 RAG
                result = await asyncio.to_thread(
                    search_docs, args["query"], int(args.get("top_k", 3))
                )
            else:  # MCP 工具（动态发现，不假设具体名字）
                result = (await session.call_tool(name, args)).content[0].text
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
    return "抱歉，我没能完成这个任务，请换个问法试试。"


async def main() -> None:
    if not LLM_KEY or not EMBED_KEY:
        raise SystemExit("缺少 deepseek_key / api_key，请写入 .env")

    async with streamable_http_client(MCP_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            discovered = await session.list_tools()
            mcp_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in discovered.tools
            ]
            tools = LOCAL_TOOLS + mcp_tools
            system = BASE_SYSTEM + "\n工具列表:\n" + "\n".join(
                f"- {t['function']['name']}: {t['function']['description']}"
                for t in tools
            )

            history = load_memory()

            async def ask(user_input: str) -> str:
                messages = [{"role": "system", "content": system}] + history
                messages.append({"role": "user", "content": user_input})
                answer = await react(messages, tools, session)
                save_memory(user_input, answer)
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": answer})
                return answer

            print("旅游助手就绪，输入 q/exit 退出")
            while True:
                text = input("\n你: ").strip()
                if not text:
                    continue
                if text.lower() in {"q", "exit", "quit"}:
                    break
                print(f"\n助手: {await ask(text)}")


if __name__ == "__main__":
    asyncio.run(main())
