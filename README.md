# agent-demo

RAG 文档检索 + 高德天气 MCP Server 的 Agent 示例项目。

## 运行环境（唯一使用此环境）

| 项 | 值 |
|---|---|
| 环境管理器 | conda（`/Users/d/miniconda3`，已配清华源） |
| 环境名 | `agent-demo` |
| Python | 3.11.13（`/Users/d/miniconda3/envs/agent-demo`） |
| pip 源 | 阿里云 `https://mirrors.aliyun.com/pypi/simple/` |

```bash
conda activate agent-demo
python scripts/xxx.py
# 或直接用绝对路径：/Users/d/miniconda3/envs/agent-demo/bin/python scripts/xxx.py
```

> 项目根目录曾有用 uv 创建的 `.venv`，已废弃删除，不要再创建。

## 依赖

```bash
pip install "chromadb>=1.0.0" "httpx>=0.27.0" "langchain-text-splitters>=1.1.3" "mcp>=1.0.0" "python-dotenv>=1.0.0" "onnxruntime<1.24"
conda install -c conda-forge cryptography   # 必须：pip 在 Intel Mac 无 wheel，需 Rust 编译
```

| 包 | 版本 | 说明 |
|---|---|---|
| mcp | 2.3.0 | v2 中 `FastMCP` 已改名 `MCPServer` |
| chromadb | 1.5.9 | 向量库，集合名 `knowledge`，cosine |
| onnxruntime | 1.23.2 | Intel Mac 必须 `<1.24`（新版无 x86_64 wheel） |

## 配置（.env）

```bash
api_key=sk-...          # 千问开放平台（DashScope），embedding 用
AMAP_API_KEY=xxx        # 高德开放平台，天气接口用
```

## 脚本

| 脚本 | 作用 |
|---|---|
| `scripts/ingest.py` | 读 `data/doc` → 语义切片（250/50）→ qwen3-vl-embedding → 中间结果落盘 → 写入 Chroma。`--reset` 清空重建，`--dry-run` 只看切片 |
| `scripts/search.py` | 检索自检：query → embedding → Chroma top-k |
| `mcp_server/server.py` | 高德天气 MCP Server（独立进程，仅天气工具）。`python -m mcp_server.server` stdio；`--transport http [--port 9000]` HTTP |
| `scripts/agent.py` | 旅游助手 ReAct Agent：**MCP Client** 连接独立 server（`MCP_SERVER_URL`，默认 `http://127.0.0.1:8000/mcp`），动态发现工具（不硬编码工具名）+ 本地 RAG + 多轮记忆。`--query` 单次 / 无参交互式 |
| `scripts/test_mcp_weather.py` | 验证天气工具：工具发现 + 实时/预报/无效城市 |

## 数据目录

```
data/doc/            原始文档（txt/md），按需新增
data/intermediate/   中间结果：chunks.jsonl、embeddings.jsonl（导入 Chroma 前必落盘）
data/chroma/         Chroma 持久化（gitignore）
```

## 常用命令

```bash
# 导入文档（切片+embedding+写库）
python scripts/ingest.py --reset

# 检索
python scripts/search.py 黄果树瀑布怎么安排

# MCP server 独立运行（HTTP，agent 通过 MCP_SERVER_URL 连接）
python -m mcp_server.server --transport http --port 8000

# 测试天气工具（stdio）
python scripts/test_mcp_weather.py

# 旅游助手 Agent（需先起 HTTP server；交互式 / 单次）
python scripts/agent.py
python scripts/agent.py --query "黄果树怎么安排，天气如何"
```

## 关键技术点

- `qwen3-vl-embedding` 不支持 OpenAI 兼容模式，必须走 DashScope 原生多模态接口 `https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding`，维度按请求指定（用 1024）
- 切片用 `RecursiveCharacterTextSplitter`，分隔符优先级：段落 → 换行 → 。！？； → ， → 兜底按字符
