"""
TTD-DR Agent - Test-Time Diffusion Deep Researcher
Implements the complete TTD-DR algorithm from the paper.
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from ttd_dr.planner.planner import ResearchPlanner
from ttd_dr.memory.state import AgentState
from ttd_dr.refinement.evaluator import LLMEvaluator
from ttd_dr.refinement.self_evolution import SelfEvolution
from ttd_dr.tools.tools import web_search_tool
from ttd_dr.retrieval.retriever import ChromaRetriever


class TTDDRAgent:
    """Test-Time Diffusion Deep Researcher for feasibility studies."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_search_steps: int = 20,
        use_self_evolution: bool = True,
        use_diffusion: bool = True
    ):
        self.model = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.planner = ResearchPlanner(self.model)
        self.evaluator = LLMEvaluator(self.model)
        self.self_evolution = SelfEvolution(self.model, self.evaluator)
        
        self.max_search_steps = max_search_steps
        self.use_self_evolution = use_self_evolution
        self.use_diffusion = use_diffusion
        
        try:
            self.retriever = ChromaRetriever()
        except FileNotFoundError:
            self.retriever = None
    
    def run(self, address: str, brief: str = "") -> AgentState:
        """Execute complete TTD-DR pipeline."""
        
        state = AgentState(query=address, brief=brief)
        
        print(f"\n🎯 Starting TTD-DR for: {address}")
        if brief:
            print(f"📋 Brief: {brief}")
        
        state.plan = self.planner.generate_plan(address, brief)
        print(f"\n📝 Generated plan with {len(state.plan.get('sections', []))} sections")
        
        if self.use_diffusion:
            state.draft_report = self._generate_initial_draft(state)
            print(f"\n✍️  Initial draft generated ({len(state.draft_report)} chars)")
        
        self._iterative_search_and_refine(state)
        
        state.final_report = self._generate_final_report(state)
        print(f"\n✅ Final report generated ({len(state.final_report)} chars)")
        
        return state
    
    def _generate_initial_draft(self, state: AgentState) -> str:
        """Generate initial noisy draft from LLM's internal knowledge."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Generate an initial draft feasibility study report.
Use your internal knowledge to create a preliminary structure.
This draft will be refined through iterative research."""),
            ("user", """Address: {address}
Brief: {brief}

Plan:
{plan}

Generate an initial draft report covering the key sections.""")
        ])
        
        chain = prompt | self.model
        response = chain.invoke({
            "address": state.query,
            "brief": state.brief or "General feasibility assessment",
            "plan": self._format_plan(state.plan)
        })
        
        return response.content
    
    def _iterative_search_and_refine(self, state: AgentState):
        """Stage 2: Iterative search with denoising."""
        print(f"\n🔍 Starting iterative search (max {self.max_search_steps} steps)...")
        
        for step in range(self.max_search_steps):
            print(f"\n  Step {step + 1}/{self.max_search_steps}")
            
            question = self._generate_search_question(state)
            if not question or question == "DONE":
                print("  ✓ Research complete")
                break
            
            print(f"  Q: {question[:80]}...")
            
            answer = self._search_and_answer(question, state)
            
            if self.use_self_evolution and step % 3 == 0:
                print("  🧬 Applying self-evolution...")
                answer = self.self_evolution.evolve_answer(question, answer, num_variants=2)
            
            state.add_search_result(question, answer)
            print(f"  A: {answer[:80]}...")
            
            if self.use_diffusion:
                state.update_draft(self._denoise_draft(state))
                print(f"  📝 Draft updated (revision {state.revision_count})")
        
        print(f"\n✅ Completed {len(state.search_history)} search iterations")
    
    def _generate_search_question(self, state: AgentState) -> str:
        """Generate next search question based on plan and context."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Generate the next specific search question for the feasibility study.

Consider:
1. The research plan sections
2. Previous questions asked
3. Current draft gaps (if available)

Return a specific, focused question or "DONE" if research is complete."""),
            ("user", """Address: {address}

Plan:
{plan}

Previous Questions:
{previous_questions}

Current Draft:
{draft}

Generate the next search question.""")
        ])
        
        previous_q = "\n".join([
            f"{i+1}. {r.question}"
            for i, r in enumerate(state.search_history[-5:])
        ]) or "None yet"
        
        chain = prompt | self.model
        response = chain.invoke({
            "address": state.query,
            "plan": self._format_plan(state.plan),
            "previous_questions": previous_q,
            "draft": state.draft_report[:500] if state.draft_report else "No draft yet"
        })
        
        return response.content.strip()
    
    def _search_and_answer(self, question: str, state: AgentState) -> str:
        """Search and synthesize answer using RAG."""
        
        web_results = web_search_tool.invoke({"query": question})
        
        kb_results = ""
        if self.retriever:
            try:
                kb_docs = self.retriever.retrieve(question, top_k=2)
                kb_results = "\n\n".join([
                    f"[{d['metadata']['name']}]\n{d['content'][:300]}"
                    for d in kb_docs
                ])
            except:
                pass
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Synthesize a comprehensive answer from the search results.
Focus on facts relevant to property feasibility analysis."""),
            ("user", """Question: {question}

Web Search Results:
{web_results}

Knowledge Base:
{kb_results}

Provide a concise, factual answer.""")
        ])
        
        chain = prompt | self.model
        response = chain.invoke({
            "question": question,
            "web_results": str(web_results)[:2000],
            "kb_results": kb_results[:1000] if kb_results else "Not available"
        })
        
        return response.content
    
    def _denoise_draft(self, state: AgentState) -> str:
        """Denoise/refine draft with new information."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Refine the draft report by incorporating new research findings.
Update sections with new facts, verify existing information, add missing details."""),
            ("user", """Current Draft:
{draft}

Latest Research:
{latest_research}

Refine the draft with this new information.""")
        ])
        
        latest = state.search_history[-3:] if state.search_history else []
        latest_text = "\n\n".join([
            f"Q: {r.question}\nA: {r.answer}"
            for r in latest
        ])
        
        chain = prompt | self.model
        response = chain.invoke({
            "draft": state.draft_report,
            "latest_research": latest_text
        })
        
        return response.content
    
    def _generate_final_report(self, state: AgentState) -> str:
        """Stage 3: Generate final comprehensive report."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Generate a comprehensive, investor-grade feasibility study report.

Structure:
1. Executive Summary
2. Site Context & Location
3. Zoning & Regulatory Analysis
4. Environmental Constraints
5. Infrastructure & Utilities
6. Market Analysis
7. Development Opportunities
8. Risks & Challenges
9. Recommendations

Use professional language, cite sources, be factual."""),
            ("user", """Address: {address}
Brief: {brief}

Research Findings:
{research_findings}

Current Draft:
{draft}

Generate the final comprehensive feasibility study report.""")
        ])
        
        research_text = "\n\n".join([
            f"Q: {r.question}\nA: {r.answer}"
            for r in state.search_history
        ])
        
        chain = prompt | self.model
        response = chain.invoke({
            "address": state.query,
            "brief": state.brief or "General feasibility assessment",
            "research_findings": research_text,
            "draft": state.draft_report if self.use_diffusion else ""
        })
        
        return response.content
    
    def _format_plan(self, plan: dict) -> str:
        """Format plan for display."""
        sections = plan.get("sections", [])
        lines = []
        for i, section in enumerate(sections, 1):
            lines.append(f"{i}. {section.get('title', 'Section')}")
            for q in section.get("questions", [])[:3]:
                lines.append(f"   - {q}")
        return "\n".join(lines)


# ============================================================================
# LangGraph Integration - Following LangGraph Quickstart Best Practices
# ============================================================================

from typing import TypedDict, Annotated, Sequence, Dict, Any, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
import operator


class TTDDRGraphState(TypedDict):
    """State for TTD-DR agent graph."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    address: str
    brief: str
    plan: Dict[str, Any]
    search_history: list
    draft_report: str
    final_report: str
    step_count: int


# Initialize components (lazy loading for retriever)
_model = None
_planner = None
_evaluator = None
_self_evolution = None
_retriever = None


def _get_components():
    """Lazy initialization of components."""
    global _model, _planner, _evaluator, _self_evolution, _retriever
    
    if _model is None:
        _model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        _planner = ResearchPlanner(_model)
        _evaluator = LLMEvaluator(_model)
        _self_evolution = SelfEvolution(_model, _evaluator)
        
        try:
            _retriever = ChromaRetriever()
        except FileNotFoundError:
            _retriever = None
    
    return _model, _planner, _evaluator, _self_evolution, _retriever


# ============================================================================
# Node Definitions
# ============================================================================

def parse_input_node(state: TTDDRGraphState) -> dict:
    """Parse user input and initialize state."""
    messages = state.get("messages", [])
    
    if not messages:
        return {"address": "", "brief": "", "search_history": [], "step_count": 0}
    
    last_msg = messages[-1]
    
    if isinstance(last_msg, dict):
        user_input = last_msg.get("content", "")
    else:
        user_input = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
    
    return {
        "address": user_input,
        "brief": "",
        "search_history": [],
        "step_count": 0,
        "messages": [AIMessage(content=f"🎯 Starting TTD-DR for: {user_input}")]
    }


def stage1_plan_node(state: TTDDRGraphState) -> dict:
    """Stage 1: Generate structured research plan."""
    _, planner, _, _, _ = _get_components()
    
    plan = planner.generate_plan(state["address"], state.get("brief", ""))
    
    return {
        "plan": plan,
        "messages": [AIMessage(content=f"📝 Stage 1 Complete: Generated plan with {len(plan.get('sections', []))} sections")]
    }


def stage2_draft_node(state: TTDDRGraphState) -> dict:
    """Stage 2a: Generate initial noisy draft (diffusion start)."""
    model, _, _, _, _ = _get_components()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are generating an initial draft feasibility study report.
Use your internal knowledge to create a preliminary structure.
This draft will be refined through iterative research and denoising."""),
        ("user", """Address: {address}
Brief: {brief}

Research Plan:
{plan}

Generate an initial draft report covering the key sections outlined in the plan.""")
    ])
    
    plan_text = "\n".join([
        f"{i+1}. {s.get('title', 'Section')}"
        for i, s in enumerate(state["plan"].get("sections", []))
    ])
    
    chain = prompt | model
    response = chain.invoke({
        "address": state["address"],
        "brief": state.get("brief", "General feasibility assessment"),
        "plan": plan_text
    })
    
    return {
        "draft_report": response.content,
        "messages": [AIMessage(content=f"✍️ Diffusion Start: Initial draft generated ({len(response.content)} chars)")]
    }


def search_node(state: TTDDRGraphState) -> dict:
    """Stage 2b: Iterative search with self-evolution."""
    model, _, _, self_evolution, retriever = _get_components()
    
    step = state["step_count"]
    
    # Generate search question
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Generate the next specific, focused search question for the feasibility study.

Consider:
1. The research plan sections
2. Previous questions asked
3. Gaps in current knowledge

Return a specific question or "DONE" if research is complete."""),
        ("user", """Address: {address}
Step: {step}/2

Research Plan:
{plan}

Previous Questions:
{history}

Generate the next search question.""")
    ])
    
    history = "\n".join([
        f"{i+1}. {q}"
        for i, (q, _) in enumerate(state["search_history"][-3:])
    ]) if state["search_history"] else "None yet"
    
    plan_text = "\n".join([
        f"• {s.get('title', 'Section')}"
        for s in state["plan"].get("sections", [])
    ])
    
    chain = prompt | model
    response = chain.invoke({
        "address": state["address"],
        "step": step + 1,
        "plan": plan_text,
        "history": history
    })
    
    question = response.content.strip()
    
    if "DONE" in question.upper() or step >= 2:
        return {
            "messages": [AIMessage(content="✓ Research phase complete")]
        }
    
    # Perform search
    web_results = web_search_tool.invoke({"query": question})
    
    kb_results = ""
    if retriever:
        try:
            kb_docs = retriever.retrieve(question, top_k=2)
            kb_results = "\n\n".join([
                f"[{d['metadata']['name']}]\n{d['content'][:300]}"
                for d in kb_docs
            ])
        except:
            pass
    
    # Synthesize answer
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "Synthesize a comprehensive answer from the search results. Focus on facts relevant to property feasibility."),
        ("user", "Question: {q}\n\nWeb Results:\n{web}\n\nKnowledge Base:\n{kb}\n\nProvide a concise, factual answer.")
    ])
    
    chain = answer_prompt | model
    answer_resp = chain.invoke({
        "q": question,
        "web": str(web_results)[:1500],
        "kb": kb_results[:500] if kb_results else "Not available"
    })
    
    answer = answer_resp.content
    
    # Apply self-evolution every 3 steps
    messages_update = [AIMessage(content=f"🔍 Step {step+1}: {question[:80]}...")]
    
    if step % 2 == 0 and step > 0:
        messages_update.append(AIMessage(content="🧬 Self-Evolution: Generating variants and selecting best answer..."))
        answer = self_evolution.evolve_answer(question, answer, num_variants=2, num_iterations=1)
    
    # Update search history
    new_history = state["search_history"] + [(question, answer)]
    
    return {
        "search_history": new_history,
        "step_count": step + 1,
        "messages": messages_update
    }


def denoise_node(state: TTDDRGraphState) -> dict:
    """Stage 2c: Denoise draft with new research (diffusion denoising step)."""
    model, _, _, _, _ = _get_components()
    
    if not state["search_history"]:
        return {"messages": [AIMessage(content="⚠️ No research to denoise with")]}
    
    # Get latest research findings
    latest = state["search_history"][-2:] if len(state["search_history"]) >= 2 else state["search_history"]
    latest_text = "\n\n".join([
        f"Q: {q}\nA: {a}"
        for q, a in latest
    ])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are refining a draft feasibility study report through diffusion-style denoising.

Your task:
1. Incorporate the new research findings into the draft
2. Update sections with new facts and data
3. Verify and correct any inconsistencies
4. Maintain the overall structure
5. Improve clarity and coherence

This is an iterative refinement process - each step should make the draft more accurate and comprehensive."""),
        ("user", """Current Draft:
{draft}

New Research Findings:
{research}

Refine the draft by incorporating these findings.""")
    ])
    
    chain = prompt | model
    response = chain.invoke({
        "draft": state["draft_report"][:1500],
        "research": latest_text
    })
    
    revision_num = state["step_count"]
    
    return {
        "draft_report": response.content,
        "messages": [AIMessage(content=f"📝 Denoising: Draft refined with latest research (revision {revision_num})")]
    }


def final_report_node(state: TTDDRGraphState) -> dict:
    """Stage 3: Generate final comprehensive report."""
    model, _, _, _, _ = _get_components()
    
    research_text = "\n\n".join([
        f"Q: {q}\nA: {a}"
        for q, a in state["search_history"]
    ])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Generate a comprehensive, investor-grade feasibility study report.

Structure:
1. Executive Summary
2. Site Context & Location Analysis
3. Zoning & Regulatory Framework
4. Environmental Constraints & Considerations
5. Infrastructure & Utilities Assessment
6. Market Analysis & Demographics
7. Development Opportunities
8. Risks & Challenges
9. Recommendations & Next Steps

Use professional language, cite sources where applicable, and be factual and thorough."""),
        ("user", """Address: {address}
Brief: {brief}

Research Findings:
{research}

Current Draft (from denoising):
{draft}

Generate the final comprehensive feasibility study report in markdown format.""")
    ])
    
    chain = prompt | model
    response = chain.invoke({
        "address": state["address"],
        "brief": state.get("brief", "General feasibility assessment"),
        "research": research_text[:3000],
        "draft": state["draft_report"][:1500]
    })
    
    return {
        "final_report": response.content,
        "messages": [AIMessage(content=f"✅ Stage 3 Complete: Final report generated ({len(response.content)} chars)\n\n{response.content[:500]}...")]
    }


# ============================================================================
# Routing Logic
# ============================================================================

def should_continue_search(state: TTDDRGraphState) -> Literal["denoise", "final_report"]:
    """Decide whether to continue search loop or move to final report."""
    if state["step_count"] >= 2:
        return "final_report"
    
    # Check if last search indicated DONE
    if state["messages"] and "complete" in state["messages"][-1].content.lower():
        return "final_report"
    
    return "denoise"


def after_denoise_routing(state: TTDDRGraphState) -> Literal["search", "final_report"]:
    """After denoising, decide whether to search more or finalize."""
    if state["step_count"] >= 2:
        return "final_report"
    
    return "search"


# ============================================================================
# Build Graph
# ============================================================================

def create_graph():
    """Create TTD-DR agent graph following LangGraph best practices.
    
    Graph structure:
        START → parse_input → stage1_plan → stage2_draft → search → denoise
                                                              ↑         ↓
                                                              └─────────┘
                                                                   (loop)
                                                                    ↓
                                                              final_report → END
    """
    
    # Build workflow
    workflow = StateGraph(TTDDRGraphState)
    
    # Add nodes
    workflow.add_node("parse_input", parse_input_node)
    workflow.add_node("stage1_plan", stage1_plan_node)
    workflow.add_node("stage2_draft", stage2_draft_node)
    workflow.add_node("search", search_node)
    workflow.add_node("denoise", denoise_node)
    workflow.add_node("final_report", final_report_node)
    
    # Add edges to connect nodes
    workflow.add_edge(START, "parse_input")
    workflow.add_edge("parse_input", "stage1_plan")
    workflow.add_edge("stage1_plan", "stage2_draft")
    workflow.add_edge("stage2_draft", "search")
    
    # Conditional edge after search: denoise or finalize
    workflow.add_conditional_edges(
        "search",
        should_continue_search,
        {
            "denoise": "denoise",
            "final_report": "final_report"
        }
    )
    
    # Conditional edge after denoise: loop back to search or finalize
    workflow.add_conditional_edges(
        "denoise",
        after_denoise_routing,
        {
            "search": "search",
            "final_report": "final_report"
        }
    )
    
    workflow.add_edge("final_report", END)
    
    return workflow.compile()


# Export the compiled graph
graph = create_graph()

