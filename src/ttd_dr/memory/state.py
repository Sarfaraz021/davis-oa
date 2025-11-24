"""
Agent State Management - Tracks research progress and context.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class SearchResult:
    """Represents a single search question-answer pair."""
    question: str
    answer: str
    sources: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    score: float = 0.0


@dataclass
class AgentState:
    """Maintains the complete state of the research agent."""
    
    query: str
    brief: str = ""
    
    plan: Dict[str, Any] = field(default_factory=dict)
    
    search_history: List[SearchResult] = field(default_factory=list)
    
    draft_report: str = ""
    revision_count: int = 0
    
    final_report: str = ""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_search_result(self, question: str, answer: str, sources: List[str] = None):
        """Add a search result to history."""
        result = SearchResult(
            question=question,
            answer=answer,
            sources=sources or []
        )
        self.search_history.append(result)
    
    def get_search_context(self, last_n: int = 5) -> str:
        """Get formatted recent search history."""
        recent = self.search_history[-last_n:] if last_n > 0 else self.search_history
        
        context_parts = []
        for i, result in enumerate(recent, 1):
            context_parts.append(
                f"Q{i}: {result.question}\n"
                f"A{i}: {result.answer}"
            )
        
        return "\n\n".join(context_parts)
    
    def update_draft(self, new_draft: str):
        """Update the draft report and increment revision count."""
        self.draft_report = new_draft
        self.revision_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "query": self.query,
            "brief": self.brief,
            "plan": self.plan,
            "search_history": [
                {
                    "question": r.question,
                    "answer": r.answer,
                    "sources": r.sources,
                    "timestamp": r.timestamp
                }
                for r in self.search_history
            ],
            "draft_report": self.draft_report,
            "revision_count": self.revision_count,
            "final_report": self.final_report,
            "metadata": self.metadata
        }

