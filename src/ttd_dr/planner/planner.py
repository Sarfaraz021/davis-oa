"""
Research Planner - Generates structured plans for feasibility studies.
"""

from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


class ResearchPlanner:
    """Generates structured research plans for property feasibility studies."""
    
    def __init__(self, model: ChatOpenAI):
        self.model = model
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("user", "{query}")
        ])
    
    def _get_system_prompt(self) -> str:
        return """You are an expert real estate feasibility analyst. Generate a comprehensive research plan.

Your plan should outline key areas to investigate for a property feasibility study:
1. Site Context (location, parcel details, current use)
2. Zoning & Regulations (codes, restrictions, permits)
3. Environmental Constraints (flood zones, soil, contamination)
4. Infrastructure & Utilities (access, capacity, costs)
5. Market Analysis (demographics, demand, comparables)
6. Development Opportunities (highest & best use, density)
7. Risks & Challenges (constraints, timeline, costs)

Output a JSON structure with sections and specific questions to research:
{{
  "sections": [
    {{
      "title": "Section Name",
      "questions": ["Specific question 1", "Specific question 2"]
    }}
  ]
}}"""
    
    def generate_plan(self, query: str, brief: str = "") -> Dict[str, Any]:
        """Generate research plan for a feasibility study."""
        full_query = f"Address: {query}"
        if brief:
            full_query += f"\nDeveloper Brief: {brief}"
        
        chain = self.prompt | self.model
        response = chain.invoke({"query": full_query})
        
        import json
        try:
            plan = json.loads(response.content)
        except json.JSONDecodeError:
            content = response.content
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                plan = json.loads(content[start:end])
            else:
                plan = {"sections": [{"title": "General Research", "questions": [query]}]}
        
        return plan

