"""Self-Evolution Algorithm - Component-wise optimization."""

from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from .evaluator import LLMEvaluator


class SelfEvolution:
    """Implements self-evolution for component optimization."""
    
    def __init__(self, model: ChatOpenAI, evaluator: LLMEvaluator):
        self.model = model
        self.evaluator = evaluator
    
    def evolve_answer(self, question: str, initial_answer: str, num_variants: int = 3, num_iterations: int = 1) -> str:
        """Evolve answer through variants and iterations."""
        variants = self._generate_variants(question, initial_answer, num_variants)
        evolved_variants = [self._evolve_variant(question, v, num_iterations) for v in variants]
        return self._merge_variants(question, evolved_variants)
    
    def _generate_variants(self, question: str, initial_answer: str, num_variants: int) -> List[str]:
        """Generate diverse answer variants."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate diverse, comprehensive answer. Focus on different aspects than previous."),
            ("user", "Question: {question}\nInitial: {initial_answer}")
        ])
        
        return [(prompt | self.model).invoke({"question": question, "initial_answer": initial_answer}).content for _ in range(num_variants)]
    
    def _evolve_variant(self, question: str, variant: str, num_iterations: int) -> str:
        """Evolve variant through feedback iterations."""
        current = variant
        for _ in range(num_iterations):
            score, feedback = self.evaluator.evaluate_answer(question, current)
            if score >= 8.0:
                break
            current = self._revise_with_feedback(question, current, feedback)
        return current
    
    def _revise_with_feedback(self, question: str, answer: str, feedback: str) -> str:
        """Revise answer based on feedback."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Improve answer based on feedback."),
            ("user", "Question: {question}\nAnswer: {answer}\nFeedback: {feedback}\n\nImproved:")
        ])
        return (prompt | self.model).invoke({"question": question, "answer": answer, "feedback": feedback}).content
    
    def _merge_variants(self, question: str, variants: List[str]) -> str:
        """Merge evolved variants into final answer."""
        variants_text = "\n\n---\n\n".join([f"Variant {i+1}:\n{v}" for i, v in enumerate(variants)])
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Merge candidate answers into comprehensive answer. Combine best info, reconcile conflicts."),
            ("user", "Question: {question}\nCandidates:\n{variants}")
        ])
        return (prompt | self.model).invoke({"question": question, "variants": variants_text}).content

