"""
LLM-as-Judge Evaluator for quality assessment.
"""

from typing import Dict, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


class LLMEvaluator:
    """Evaluates content quality using LLM-as-judge."""
    
    def __init__(self, model: ChatOpenAI):
        self.model = model
    
    def evaluate_answer(self, question: str, answer: str) -> Tuple[float, str]:
        """Evaluate answer quality with score and feedback."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert evaluator assessing answer quality.

Rate the answer on:
1. Helpfulness: Does it address the question comprehensively?
2. Accuracy: Is the information factually sound?
3. Completeness: Are all aspects covered?

Provide:
- Score (0-10): Overall quality rating
- Feedback: Specific improvements needed

Format:
Score: X
Feedback: Your detailed feedback here"""),
            ("user", "Question: {question}\n\nAnswer: {answer}")
        ])
        
        chain = prompt | self.model
        response = chain.invoke({"question": question, "answer": answer})
        
        content = response.content
        score = self._extract_score(content)
        feedback = self._extract_feedback(content)
        
        return score, feedback
    
    def evaluate_report(self, query: str, report: str) -> Tuple[float, str]:
        """Evaluate report quality."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert evaluating feasibility study reports.

Rate on:
1. Comprehensiveness: All key areas covered?
2. Professional Quality: Investor-grade analysis?
3. Actionability: Clear insights and recommendations?

Format:
Score: X
Feedback: Your detailed feedback"""),
            ("user", "Query: {query}\n\nReport:\n{report}")
        ])
        
        chain = prompt | self.model
        response = chain.invoke({"query": query, "report": report})
        
        content = response.content
        score = self._extract_score(content)
        feedback = self._extract_feedback(content)
        
        return score, feedback
    
    def _extract_score(self, content: str) -> float:
        """Extract numeric score from evaluation."""
        import re
        match = re.search(r'Score:\s*(\d+(?:\.\d+)?)', content, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return 5.0
    
    def _extract_feedback(self, content: str) -> str:
        """Extract feedback text from evaluation."""
        import re
        match = re.search(r'Feedback:\s*(.+)', content, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return content

