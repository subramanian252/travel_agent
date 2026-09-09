from langchain_tavily import TavilySearch

import os
from dotenv import load_dotenv

load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")

search = TavilySearch(api_key=tavily_api_key)

def tavily_search(query: str):
    result = search.run(query)

    final_results = []
    
    for i, r in enumerate(result["results"]):
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "")
        final_results.append(f"{title}\n{url}\n{content}")

    return "\n\n".join(final_results)


