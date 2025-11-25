"""Agent State Management - Tracks research progress."""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from datetime import datetime


@dataclass
class SearchResult:
    """Single search question-answer pair."""
    question: str
    answer: str
    sources: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    score: float = 0.0


@dataclass
class AgentState:
    """Research agent state."""
    query: str
    brief: str = ""
    plan: Dict[str, Any] = field(default_factory=dict)
    search_history: List[SearchResult] = field(default_factory=list)
    draft_report: str = ""
    revision_count: int = 0
    final_report: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_search_result(self, question: str, answer: str, sources: List[str] = None):
        """Add search result to history."""
        self.search_history.append(SearchResult(question=question, answer=answer, sources=sources or []))
    
    def get_search_context(self, last_n: int = 5) -> str:
        """Get formatted recent search history."""
        recent = self.search_history[-last_n:] if last_n > 0 else self.search_history
        return "\n\n".join([f"Q{i}: {r.question}\nA{i}: {r.answer}" for i, r in enumerate(recent, 1)])
    
    def update_draft(self, new_draft: str):
        """Update draft and increment revision count."""
        self.draft_report = new_draft
        self.revision_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "brief": self.brief,
            "plan": self.plan,
            "search_history": [{"question": r.question, "answer": r.answer, "sources": r.sources, "timestamp": r.timestamp} for r in self.search_history],
            "draft_report": self.draft_report,
            "revision_count": self.revision_count,
            "final_report": self.final_report,
            "metadata": self.metadata
        }

