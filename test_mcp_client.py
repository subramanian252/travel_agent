import os
import asyncio
import certifi
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.server.fastmcp import FastMCP
import requests

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

AVIATION_STACK_API_KEY = os.getenv("AVIATION_STACK_API_KEY")

WEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")



async def get_tavily_tools():
    client = MultiServerMCPClient(
        {
            "tavily": {
                "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}",
                "transport": "http"
            }
        }
    )
    
    tools = await client.get_tools()


    return tools


async def get_airport_tools():
    client = MultiServerMCPClient(
        {
            "Aviationstack MCP": {
                "transport": "stdio",
                "command": "uvx",
                "args": [
                    "--with",
                    "mcp<2",
                    "aviationstack-mcp"
                ],
                "env": {
                    "AVIATION_STACK_API_KEY": AVIATION_STACK_API_KEY
                }
            }
        }
    )
    
    tools = await client.get_tools()

    return tools

## custom mcp





async def get_weather_tools():

    client = MultiServerMCPClient(
        {
            "weather": {
                "transport": "stdio",
                "command": r"G:\ml projects\travel_multiagent\.venv\Scripts\python.exe",
                "args": [
                    r"G:\ml projects\travel_multiagent\weather_mcp_server.py"
                ],
                "env": {
                    "OPENWEATHER_API_KEY": WEATHER_API_KEY
                }
            }
        }
    )

    tools = await client.get_tools()


    return tools


if __name__ == "__main__":
    asyncio.run(get_weather_tools())
