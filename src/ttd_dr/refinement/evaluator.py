"""LLM-as-Judge Evaluator for quality assessment."""

import re
from typing import Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


class LLMEvaluator:
    """Evaluates content quality using LLM-as-judge."""
    
    def __init__(self, model: ChatOpenAI):
        self.model = model
    
    def evaluate_answer(self, question: str, answer: str) -> Tuple[float, str]:
        """Evaluate answer quality (score 0-10 + feedback)."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Expert evaluator. Rate answer on: Helpfulness, Accuracy, Completeness. Format: 'Score: X\nFeedback: ...'"),
            ("user", "Question: {question}\nAnswer: {answer}")
        ])
        
        response = (prompt | self.model).invoke({"question": question, "answer": answer})
        return self._extract_score(response.content), self._extract_feedback(response.content)
    
    def evaluate_report(self, query: str, report: str) -> Tuple[float, str]:
        """Evaluate report quality (comprehensiveness, professionalism, actionability)."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Expert evaluator. Rate feasibility report on: Comprehensiveness, Professional Quality, Actionability. Format: 'Score: X\nFeedback: ...'"),
            ("user", "Query: {query}\nReport:\n{report}")
        ])
        
        response = (prompt | self.model).invoke({"query": query, "report": report})
        return self._extract_score(response.content), self._extract_feedback(response.content)
    
    def _extract_score(self, content: str) -> float:
        """Extract numeric score."""
        match = re.search(r'Score:\s*(\d+(?:\.\d+)?)', content, re.IGNORECASE)
        return float(match.group(1)) if match else 5.0
    
    def _extract_feedback(self, content: str) -> str:
        """Extract feedback text."""
        match = re.search(r'Feedback:\s*(.+)', content, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else content

