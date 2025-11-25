"""Research Planner - Generates structured feasibility study plans."""

import json
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


class ResearchPlanner:
    """Generates structured research plans for property feasibility studies."""
    
    def __init__(self, model: ChatOpenAI):
        self.model = model
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "Expert real estate analyst. Generate comprehensive research plan covering: Site Context, Zoning, Environmental, Infrastructure, Market, Opportunities, Risks. Output JSON with 'sections' array containing objects with 'title' and 'questions' fields."),
            ("user", "{query}")
        ])
    
    def generate_plan(self, query: str, brief: str = "") -> Dict[str, Any]:
        """Generate research plan for feasibility study."""
        full_query = f"Address: {query}" + (f"\nBrief: {brief}" if brief else "")
        response = (self.prompt | self.model).invoke({"query": full_query})
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            content = response.content
            start, end = content.find('{'), content.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(content[start:end])
            return {"sections": [{"title": "General Research", "questions": [query]}]}

