"""
Self-Evolution Algorithm - Component-wise optimization.
"""

from typing import List, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from .evaluator import LLMEvaluator


class SelfEvolution:
    """Implements self-evolution for component optimization."""
    
    def __init__(self, model: ChatOpenAI, evaluator: LLMEvaluator):
        self.model = model
        self.evaluator = evaluator
    
    def evolve_answer(
        self,
        question: str,
        initial_answer: str,
        num_variants: int = 3,
        num_iterations: int = 1
    ) -> str:
        """Evolve answer through multiple variants and iterations."""
        
        variants = self._generate_variants(question, initial_answer, num_variants)
        
        evolved_variants = []
        for variant in variants:
            evolved = self._evolve_variant(question, variant, num_iterations)
            evolved_variants.append(evolved)
        
        final_answer = self._merge_variants(question, evolved_variants)
        
        return final_answer
    
    def _generate_variants(
        self,
        question: str,
        initial_answer: str,
        num_variants: int
    ) -> List[str]:
        """Generate multiple diverse answer variants."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Generate a diverse, comprehensive answer to the question.
Focus on different aspects or perspectives than previous answers."""),
            ("user", "Question: {question}\n\nInitial Answer: {initial_answer}")
        ])
        
        chain = prompt | self.model
        variants = []
        
        for i in range(num_variants):
            response = chain.invoke({
                "question": question,
                "initial_answer": initial_answer
            })
            variants.append(response.content)
        
        return variants
    
    def _evolve_variant(
        self,
        question: str,
        variant: str,
        num_iterations: int
    ) -> str:
        """Evolve a single variant through feedback iterations."""
        current = variant
        
        for _ in range(num_iterations):
            score, feedback = self.evaluator.evaluate_answer(question, current)
            
            if score >= 8.0:
                break
            
            current = self._revise_with_feedback(question, current, feedback)
        
        return current
    
    def _revise_with_feedback(
        self,
        question: str,
        answer: str,
        feedback: str
    ) -> str:
        """Revise answer based on feedback."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Improve the answer based on the feedback provided."),
            ("user", """Question: {question}

Current Answer: {answer}

Feedback: {feedback}

Provide an improved answer addressing the feedback.""")
        ])
        
        chain = prompt | self.model
        response = chain.invoke({
            "question": question,
            "answer": answer,
            "feedback": feedback
        })
        
        return response.content
    
    def _merge_variants(self, question: str, variants: List[str]) -> str:
        """Merge multiple evolved variants into final answer."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Merge the candidate answers into a single comprehensive answer.
Combine the best information from all variants, reconcile conflicts logically."""),
            ("user", "Question: {question}\n\nCandidate Answers:\n{variants}")
        ])
        
        variants_text = "\n\n---\n\n".join([
            f"Variant {i+1}:\n{v}" for i, v in enumerate(variants)
        ])
        
        chain = prompt | self.model
        response = chain.invoke({
            "question": question,
            "variants": variants_text
        })
        
        return response.content

