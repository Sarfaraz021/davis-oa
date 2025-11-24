"""
Minimal TTD-DR agent using LangGraph ReAct pattern.
Integrates Tavily web search and Chroma retrieval.
"""

import os
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearchResults
from langgraph.prebuilt import create_react_agent

from ..retrieval.chroma_retriever import ChromaRetriever
from ..prompts.agent_prompt import AGENT_SYSTEM_PROMPT

load_dotenv()


# Initialize Chroma retriever
_retriever = ChromaRetriever()


@tool
def retrieve_feasibility_knowledge(
    query: Annotated[str, "The feasibility research question"]
) -> str:
    """
    Retrieve curated knowledge about real estate feasibility, zoning, 
    environmental constraints, infrastructure, and development best practices.
    
    Use this tool when you need information about:
    - Zoning codes and regulations
    - Environmental hazards (flood zones, soil conditions, contamination)
    - Census demographics and market data
    - Infrastructure and utilities
    - Development constraints and opportunities
    """
    results = _retriever.retrieve(query, top_k=3)
    
    context_parts = []
    for i, result in enumerate(results, 1):
        source = result["metadata"]["source"]
        name = result["metadata"]["name"]
        content = result["content"][:500]  # Truncate for context window
        context_parts.append(
            f"[Source {i}: {name}]\n{source}\n{content}..."
        )
    
    return "\n\n".join(context_parts)


@tool
def web_search(
    query: Annotated[str, "The search query for real-time web information"]
) -> str:
    """
    Search the web for real-time information about parcels, locations, 
    recent market data, or current regulations.
    
    Use this tool when you need:
    - Current zoning or land use information for a specific address
    - Recent news about the area or development projects
    - Market trends or property values
    - Local government information or permits
    """
    tavily_tool = TavilySearchResults(
        max_results=3,
        search_depth="basic",
        include_answer=False,
        include_raw_content=False,
    )
    
    results = tavily_tool.invoke({"query": query})
    
    # Format Tavily results
    formatted = []
    for i, result in enumerate(results, 1):
        title = result.get("title", "No title")
        url = result.get("url", "")
        content = result.get("content", "")[:400]
        formatted.append(f"[Result {i}: {title}]\n{url}\n{content}...")
    
    return "\n\n".join(formatted)


def create_agent():
    """
    Create the TTD-DR ReAct agent with web search and retrieval tools.
    
    Returns:
        A LangGraph agent executor
    """
    # Initialize LLM
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    # Create ReAct agent
    agent = create_react_agent(
        model=model,
        tools=[web_search, retrieve_feasibility_knowledge],
        state_modifier=AGENT_SYSTEM_PROMPT,
    )
    
    # Add metadata
    agent.name = "TTD-DR Feasibility Agent"
    agent.description = (
        "A ReAct agent that generates feasibility study reports using "
        "web search and curated knowledge retrieval."
    )
    
    return agent


# Convenience function for direct invocation
def run_agent(user_input: str, verbose: bool = True):
    """
    Run the agent on a user query and stream results.
    
    Args:
        user_input: User query (e.g., "Analyze 123 Main St, San Francisco")
        verbose: Print intermediate steps
    
    Returns:
        Final agent response
    """
    agent = create_agent()
    
    final_response = None
    for step in agent.stream(
        {"messages": [("user", user_input)]},
        stream_mode="values",
    ):
        if verbose:
            step["messages"][-1].pretty_print()
        final_response = step["messages"][-1]
    
    return final_response
