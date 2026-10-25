#!/usr/bin/env python3
"""高德开放平台天气 MCP Server（独立进程，仅天气工具）。

运行（项目根目录）：
    python -m mcp_server.server                       # stdio（供测试脚本用）
    python -m mcp_server.server --transport http       # HTTP，默认 127.0.0.1:8000/mcp
    python -m mcp_server.server --transport http --port 9000
"""

import argparse
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("AMAP_API_KEY", "")
AMAP_URL = "https://restapi.amap.com/v3/weather/weatherInfo"

server = MCPServer(name="amap-weather")


@server.tool()
def get_weather(city: str, extensions: str = "base") -> str:
    """查询城市实时天气或未来三天预报（高德开放平台）。

    Args:
        city: 城市名称（如"北京"）或 adcode（如"110000"）
        extensions: base=实时天气，all=未来三天预报
    """
    response = httpx.get(
        AMAP_URL,
        params={"key": API_KEY, "city": city, "extensions": extensions},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    if data.get("status") != "1":
        return f"查询失败: {data.get('info')}"

    if extensions == "all":
        forecasts = data.get("forecasts") or []
        casts = (forecasts[0].get("casts") or []) if forecasts else []
        if not casts:
            return f"未查询到 {city} 的天气数据"
        return "\n".join(
            f"{c['date']} 白天:{c['dayweather']}/{c['daytemp']}℃ "
            f"夜间:{c['nightweather']}/{c['nighttemp']}℃"
            for c in casts
        )

    lives = data.get("lives") or []
    if not lives:
        return f"未查询到 {city} 的天气数据"
    lives = lives[0]
    return (
        f"{lives['province']} {lives['city']}（更新于 {lives['reporttime']}）\n"
        f"天气:{lives['weather']} 温度:{lives['temperature']}℃ "
        f"风向:{lives['winddirection']} 风力:{lives['windpower']} "
        f"湿度:{lives['humidity']}%"
    )


if __name__ == "__main__":
    if not API_KEY:
        raise SystemExit("缺少 AMAP_API_KEY，请先写入 .env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.transport == "http":
        server.run(
            transport="streamable-http",
            host=args.host,
            port=args.port,
        )
    else:
        server.run(transport="stdio")
