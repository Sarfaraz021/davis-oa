"""
Tavily Search Tools for TTD-DR Feasibility Agent
"""

from typing import Annotated
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

# Initialize Tavily Search instance
tavily_search = TavilySearch(
    max_results=2,
    topic="news",
    # include_answer=False,
    # include_raw_content=False,
    # include_images=False,
    # include_image_descriptions=False,
    # search_depth="basic",
    # time_range="day",
    # include_domains=None,
    # exclude_domains=None
)

# Export the tool for use in agents
web_search_tool = tavily_search

