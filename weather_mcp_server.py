

import os
import asyncio
import certifi
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.server.fastmcp import FastMCP
import requests
import json

WEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
mcp = FastMCP("Weather MCP Server")


@mcp.tool()
def get_weather(city: str) -> str:
    """Get weather for a city"""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    response = requests.get(url)

    data = response.json()
    

    if response.status_code == 200:
        values =  {
            "city": city,
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"],
            "condition": data["weather"][0]["description"],
            "feels_like": data["main"]["feels_like"]
        }
        return json.dumps(values)
    else:
        return "Error: " + str(response.status_code)

@mcp.tool()
def get_forecast(city: str) -> str:
    """Get weather forecast for a city"""
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={WEATHER_API_KEY}&units=metric"
    response = requests.get(url)

    data = response.json()

    forecast = []
    
    if response.status_code == 200:
        for item in data["list"][:5]:
            forecast.append({
                "datetime": item["dt_txt"],
                "temperature": item["main"]["temp"],
                "weather": item["weather"][0]["description"]
            })
        values = {
            "city": city,
            "forecast": forecast
        }
        return json.dumps(values)
    else:
        return "Error: " + str(response.status_code)


if __name__ == "__main__":
    mcp.run()