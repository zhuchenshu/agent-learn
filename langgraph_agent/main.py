#!/usr/bin/env python3
"""旅游助手 Agent（LangGraph 编排版，极简）。

与 scripts/agent.py 功能一致，但用 LangGraph 的 StateGraph 显式建模 ReAct 循环：
    START → agent（LLM 决策）→ 有工具调用则 tools（执行）→ 回 agent……直到无调用输出回答

- MCP Client：连接独立运行的天气 server（默认 http://127.0.0.1:8000/mcp）
- 本地 RAG：search_docs 直连 Chroma（不走 MCP）
- 记忆：data/memory/history.txt 追加，启动时加载最近轮次

用法：
    python langgraph_agent/main.py   # 交互式对话，q/exit 退出
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Annotated, TypedDict

import chromadb
import httpx
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

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
需要实时信息（如天气）时调用天气工具；需要本地攻略知识时调用 search_docs 并注明来源。"""

# 本地 RAG 工具（不走 MCP）
LOCAL_TOOL_SCHEMA = {
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


# ---------- 模型 ----------

llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=LLM_KEY,
    base_url="https://api.deepseek.com",
    temperature=0.3,
)


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

def load_memory() -> list:
    if not MEMORY_FILE.exists():
        return []
    out = []
    for line in MEMORY_FILE.read_text(encoding="utf-8").splitlines()[-MEMORY_LINES:]:
        if line.startswith("用户: "):
            out.append(HumanMessage(content=line[3:]))
        elif line.startswith("助手: "):
            out.append(AIMessage(content=line[3:]))
    return out


def save_memory(user_text: str, answer: str) -> None:
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with MEMORY_FILE.open("a", encoding="utf-8") as f:
        f.write(f"用户: {user_text}\n助手: {answer}\n")


# ---------- LangGraph 编排 ----------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # 消息随节点自动追加合并
    iterations: int


def make_tool_schemas(mcp_tools: list) -> list[dict]:
    """动态组装 OpenAI 格式工具 schema：本地 RAG + MCP 天气工具。"""
    mcp_schemas = [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in mcp_tools
    ]
    return [LOCAL_TOOL_SCHEMA, *mcp_schemas]


async def agent_node(state: AgentState, session: ClientSession) -> dict:
    """LLM 决策节点：带工具调用模型生成回复（可能含 tool_calls）。"""
    response = await llm.bind_tools(tool_schemas).ainvoke(state["messages"])
    return {"messages": [response], "iterations": state["iterations"] + 1}


async def tools_node(state: AgentState, session: ClientSession) -> dict:
    """工具执行节点：执行上一步 LLM 请求的所有工具调用，结果回填为 ToolMessage。"""
    last = state["messages"][-1]
    outputs = []
    for call in last.tool_calls:
        name, args = call["name"], call["args"]
        print(f"  [工具] {name}{args}")
        if name == "search_docs":  # 本地 RAG
            result = await asyncio.to_thread(
                search_docs, args.get("query", ""), int(args.get("top_k", 3))
            )
        else:  # MCP 工具（动态发现，不假设具体名字）
            result = (await session.call_tool(name, args)).content[0].text
        outputs.append(ToolMessage(content=result, tool_call_id=call["id"]))
    return {"messages": outputs}


def should_continue(state: AgentState) -> str:
    """条件路由：还有工具调用且未超次数 → 继续执行工具；否则结束。"""
    last = state["messages"][-1]
    if last.tool_calls and state["iterations"] < MAX_ITERS:
        return "tools"
    return "end"


async def main() -> None:
    if not LLM_KEY or not EMBED_KEY:
        raise SystemExit("缺少 deepseek_key / api_key，请写入 .env")

    async with streamable_http_client(MCP_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            discovered = await session.list_tools()

            # 组装工具 schema（闭包变量，节点里引用）
            global tool_schemas
            tool_schemas = make_tool_schemas(discovered.tools)

            system = SystemMessage(
                content=BASE_SYSTEM + "\n工具列表:\n" + "\n".join(
                    f"- {s['function']['name']}: {s['function']['description']}"
                    for s in tool_schemas
                )
            )

            # 图：START → agent → (有工具调用) tools → agent …… → END
            async def agent_step(s: AgentState) -> dict:
                return await agent_node(s, session)

            async def tools_step(s: AgentState) -> dict:
                return await tools_node(s, session)

            builder = StateGraph(AgentState)
            builder.add_node("agent", agent_step)
            builder.add_node("tools", tools_step)
            builder.add_edge(START, "agent")
            builder.add_conditional_edges(
                "agent", should_continue, {"tools": "tools", "end": END}
            )
            builder.add_edge("tools", "agent")
            graph = builder.compile()

            history = load_memory()

            async def ask(user_input: str) -> str:
                state = await graph.ainvoke(
                    {
                        "messages": [system, *history, HumanMessage(content=user_input)],
                        "iterations": 0,
                    }
                )
                answer = state["messages"][-1].content.strip()
                save_memory(user_input, answer)
                history.append(HumanMessage(content=user_input))
                history.append(AIMessage(content=answer))
                return answer

            print("旅游助手（LangGraph 版）就绪，输入 q/exit 退出")
            while True:
                text = input("\n你: ").strip()
                if not text:
                    continue
                if text.lower() in {"q", "exit", "quit"}:
                    break
                print(f"\n助手: {await ask(text)}")


if __name__ == "__main__":
    asyncio.run(main())
