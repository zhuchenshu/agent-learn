#!/usr/bin/env python3
"""测试天气 MCP 工具：工具发现 + 实时/预报/异常参数。

用法：python scripts/test_mcp_weather.py
"""

import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
SERVER_ARGS = ["-m", "mcp_server.server"]

PASS, FAIL = "✓", "✗"


async def main() -> None:
    params = StdioServerParameters(command=PYTHON, args=SERVER_ARGS, cwd=str(ROOT))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1. 工具发现
            tools = (await session.list_tools()).tools
            names = [t.name for t in tools]
            print(f"{PASS if 'get_weather' in names else FAIL} 工具发现: {names}")

            # 2. 实时天气（城市名）
            result = await session.call_tool("get_weather", {"city": "北京"})
            text = result.content[0].text if result.content else str(result)
            assert "北京" in text and "℃" in text, f"实时天气异常: {text}"
            print(f"{PASS} 实时天气(北京):\n{text}")

            # 3. 三天预报（adcode）
            result = await session.call_tool(
                "get_weather", {"city": "110000", "extensions": "all"}
            )
            text = result.content[0].text if result.content else str(result)
            assert "白天" in text and "夜间" in text, f"预报异常: {text}"
            print(f"{PASS} 三天预报(110000):\n{text}")

            # 4. 不存在的城市 → 返回错误信息而不是崩溃
            result = await session.call_tool("get_weather", {"city": "不存在的地方"})
            text = result.content[0].text if result.content else str(result)
            print(f"{'✓' if '失败' in text or 'OK' not in text else '✗'} 无效城市: {text}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"{FAIL} 测试失败: {exc}")
        raise SystemExit(1)
