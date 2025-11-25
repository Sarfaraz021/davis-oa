"""Tavily Search Tools for TTD-DR Feasibility Agent."""

from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

web_search_tool = TavilySearch(max_results=2, topic="news")

