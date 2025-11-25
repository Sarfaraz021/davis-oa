"""TTD-DR Agent - Test-Time Diffusion Deep Researcher."""

import os
import sys
from pathlib import Path
from typing import Optional

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
    """Implements Test-Time Diffusion Deep Researcher for feasibility studies."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_search_steps: int = 20,
        use_self_evolution: bool = True,
        use_diffusion: bool = True
    ):
        self.model = ChatOpenAI(model=model_name, temperature=temperature, openai_api_key=os.getenv("OPENAI_API_KEY"))
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
        """Execute TTD-DR pipeline: plan → draft → iterate → finalize."""
        state = AgentState(query=address, brief=brief)
        
        print(f"\n🎯 Starting TTD-DR for: {address}")
        if brief:
            print(f"📋 Brief: {brief}")
        
        state.plan = self.planner.generate_plan(address, brief)
        print(f"\n📝 Generated {len(state.plan.get('sections', []))} research sections")
        
        if self.use_diffusion:
            state.draft_report = self._generate_initial_draft(state)
            print(f"\n✍️  Initial draft: {len(state.draft_report)} chars")
        
        self._iterative_search_and_refine(state)
        state.final_report = self._generate_final_report(state)
        print(f"\n✅ Final report: {len(state.final_report)} chars")
        
        return state
    
    def _generate_initial_draft(self, state: AgentState) -> str:
        """Generate initial draft from LLM's internal knowledge (diffusion start)."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate an initial draft feasibility study. Use internal knowledge to create a preliminary structure that will be refined through research."),
            ("user", "Address: {address}\nBrief: {brief}\nPlan:\n{plan}\n\nGenerate initial draft.")
        ])
        
        response = (prompt | self.model).invoke({
            "address": state.query,
            "brief": state.brief or "General feasibility assessment",
            "plan": self._format_plan(state.plan)
        })
        return response.content
    
    def _iterative_search_and_refine(self, state: AgentState):
        """Iterative search with self-evolution and denoising."""
        print(f"\n🔍 Iterative search (max {self.max_search_steps} steps)")
        
        for step in range(self.max_search_steps):
            print(f"\n  Step {step + 1}/{self.max_search_steps}")
            
            question = self._generate_search_question(state)
            if not question or question == "DONE":
                print("  ✓ Research complete")
                break
            
            print(f"  Q: {question[:80]}...")
            answer = self._search_and_answer(question, state)
            
            if self.use_self_evolution and step % 3 == 0:
                print("  🧬 Self-evolution")
                answer = self.self_evolution.evolve_answer(question, answer, num_variants=2)
            
            state.add_search_result(question, answer)
            print(f"  A: {answer[:80]}...")
            
            if self.use_diffusion:
                state.update_draft(self._denoise_draft(state))
                print(f"  📝 Denoised (revision {state.revision_count})")
        
        print(f"\n✅ Completed {len(state.search_history)} iterations")
    
    def _generate_search_question(self, state: AgentState) -> str:
        """Generate next focused search question."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate the next specific search question for the feasibility study. Consider the plan, previous questions, and current gaps. Return a focused question or 'DONE' if complete."),
            ("user", "Address: {address}\nPlan:\n{plan}\nPrevious:\n{previous}\nDraft:\n{draft}\n\nNext question:")
        ])
        
        previous = "\n".join([f"{i+1}. {r.question}" for i, r in enumerate(state.search_history[-5:])]) or "None"
        
        response = (prompt | self.model).invoke({
            "address": state.query,
            "plan": self._format_plan(state.plan),
            "previous": previous,
            "draft": state.draft_report[:500] if state.draft_report else "No draft"
        })
        return response.content.strip()
    
    def _search_and_answer(self, question: str, state: AgentState) -> str:
        """Search and synthesize answer using web and knowledge base."""
        web_results = web_search_tool.invoke({"query": question})
        
        kb_results = ""
        if self.retriever:
            try:
                kb_docs = self.retriever.retrieve(question, top_k=2)
                kb_results = "\n\n".join([f"[{d['metadata']['name']}]\n{d['content'][:300]}" for d in kb_docs])
            except:
                pass
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Synthesize a comprehensive answer from search results. Focus on property feasibility facts."),
            ("user", "Question: {question}\n\nWeb:\n{web}\n\nKB:\n{kb}\n\nAnswer:")
        ])
        
        response = (prompt | self.model).invoke({
            "question": question,
            "web": str(web_results)[:2000],
            "kb": kb_results[:1000] if kb_results else "N/A"
        })
        return response.content
    
    def _denoise_draft(self, state: AgentState) -> str:
        """Refine draft by incorporating latest research findings."""
        latest = state.search_history[-3:] if state.search_history else []
        latest_text = "\n\n".join([f"Q: {r.question}\nA: {r.answer}" for r in latest])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Refine the draft by incorporating new research. Update facts, verify information, add details."),
            ("user", "Draft:\n{draft}\n\nLatest:\n{latest}\n\nRefine:")
        ])
        
        response = (prompt | self.model).invoke({"draft": state.draft_report, "latest": latest_text})
        return response.content
    
    def _generate_final_report(self, state: AgentState) -> str:
        """Generate final comprehensive report with all research findings."""
        research_text = "\n\n".join([f"Q: {r.question}\nA: {r.answer}" for r in state.search_history])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate a comprehensive investor-grade feasibility study. Structure: Executive Summary, Site Context, Zoning, Environmental, Infrastructure, Market, Opportunities, Risks, Recommendations. Use professional language and cite sources."),
            ("user", "Address: {address}\nBrief: {brief}\nResearch:\n{research}\nDraft:\n{draft}\n\nGenerate final report:")
        ])
        
        response = (prompt | self.model).invoke({
            "address": state.query,
            "brief": state.brief or "General feasibility assessment",
            "research": research_text,
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
# LangGraph Integration
# ============================================================================

from typing import TypedDict, Annotated, Sequence, Dict, Any, Literal
from langchain_core.messages import BaseMessage, AIMessage
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


_model, _planner, _evaluator, _self_evolution, _retriever = None, None, None, None, None


def _get_components():
    """Lazy initialization of components."""
    global _model, _planner, _evaluator, _self_evolution, _retriever
    
    if _model is None:
        _model = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
        _planner = ResearchPlanner(_model)
        _evaluator = LLMEvaluator(_model)
        _self_evolution = SelfEvolution(_model, _evaluator)
        
        try:
            _retriever = ChromaRetriever()
        except FileNotFoundError:
            _retriever = None
    
    return _model, _planner, _evaluator, _self_evolution, _retriever


# ============================================================================
# Graph Nodes
# ============================================================================

def parse_input_node(state: TTDDRGraphState) -> dict:
    """Parse input and initialize state."""
    messages = state.get("messages", [])
    if not messages:
        return {"address": "", "brief": "", "search_history": [], "step_count": 0}
    
    last_msg = messages[-1]
    user_input = last_msg.get("content", "") if isinstance(last_msg, dict) else (last_msg.content if hasattr(last_msg, 'content') else str(last_msg))
    
    return {
        "address": user_input,
        "brief": "",
        "search_history": [],
        "step_count": 0,
        "messages": [AIMessage(content=f"🎯 Starting: {user_input}")]
    }


def stage1_plan_node(state: TTDDRGraphState) -> dict:
    """Generate research plan."""
    _, planner, _, _, _ = _get_components()
    plan = planner.generate_plan(state["address"], state.get("brief", ""))
    return {
        "plan": plan,
        "messages": [AIMessage(content=f"📝 Plan: {len(plan.get('sections', []))} sections")]
    }


def stage2_draft_node(state: TTDDRGraphState) -> dict:
    """Generate initial draft (diffusion start)."""
    model, _, _, _, _ = _get_components()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate an initial feasibility draft. Use internal knowledge to create preliminary structure that will be refined through research."),
        ("user", "Address: {address}\nBrief: {brief}\nPlan:\n{plan}\n\nGenerate initial draft.")
    ])
    
    plan_text = "\n".join([f"{i+1}. {s.get('title', 'Section')}" for i, s in enumerate(state["plan"].get("sections", []))])
    
    response = (prompt | model).invoke({
        "address": state["address"],
        "brief": state.get("brief", "General feasibility assessment"),
        "plan": plan_text
    })
    
    return {
        "draft_report": response.content,
        "messages": [AIMessage(content=f"✍️ Initial draft: {len(response.content)} chars")]
    }


def search_node(state: TTDDRGraphState) -> dict:
    """Perform iterative search with self-evolution."""
    model, _, _, self_evolution, retriever = _get_components()
    step = state["step_count"]
    
    # Generate question
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate the next specific search question for feasibility study. Consider plan, previous questions, and knowledge gaps. Return focused question or 'DONE'."),
        ("user", "Address: {address}\nStep: {step}/2\nPlan:\n{plan}\nPrevious:\n{history}\n\nNext question:")
    ])
    
    history = "\n".join([f"{i+1}. {q}" for i, (q, _) in enumerate(state["search_history"][-3:])]) if state["search_history"] else "None"
    plan_text = "\n".join([f"• {s.get('title', 'Section')}" for s in state["plan"].get("sections", [])])
    
    response = (prompt | model).invoke({"address": state["address"], "step": step + 1, "plan": plan_text, "history": history})
    question = response.content.strip()
    
    if "DONE" in question.upper() or step >= 2:
        return {"messages": [AIMessage(content="✓ Research complete")]}
    
    # Search
    web_results = web_search_tool.invoke({"query": question})
    kb_results = ""
    if retriever:
        try:
            kb_docs = retriever.retrieve(question, top_k=2)
            kb_results = "\n\n".join([f"[{d['metadata']['name']}]\n{d['content'][:300]}" for d in kb_docs])
        except:
            pass
    
    # Synthesize answer
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "Synthesize answer from search results. Focus on property feasibility facts."),
        ("user", "Question: {q}\nWeb:\n{web}\nKB:\n{kb}\n\nAnswer:")
    ])
    
    answer = (answer_prompt | model).invoke({
        "q": question,
        "web": str(web_results)[:1500],
        "kb": kb_results[:500] if kb_results else "N/A"
    }).content
    
    # Self-evolution
    messages_update = [AIMessage(content=f"🔍 Step {step+1}: {question[:80]}...")]
    if step % 2 == 0 and step > 0:
        messages_update.append(AIMessage(content="🧬 Self-evolution"))
        answer = self_evolution.evolve_answer(question, answer, num_variants=2, num_iterations=1)
    
    return {
        "search_history": state["search_history"] + [(question, answer)],
        "step_count": step + 1,
        "messages": messages_update
    }


def denoise_node(state: TTDDRGraphState) -> dict:
    """Denoise draft with new research (diffusion denoising)."""
    model, _, _, _, _ = _get_components()
    
    if not state["search_history"]:
        return {"messages": [AIMessage(content="⚠️ No research")]}
    
    latest = state["search_history"][-2:] if len(state["search_history"]) >= 2 else state["search_history"]
    latest_text = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in latest])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Refine draft through diffusion denoising. Incorporate research, update facts, verify consistency, maintain structure, improve clarity."),
        ("user", "Draft:\n{draft}\n\nResearch:\n{research}\n\nRefine:")
    ])
    
    response = (prompt | model).invoke({"draft": state["draft_report"][:1500], "research": latest_text})
    
    return {
        "draft_report": response.content,
        "messages": [AIMessage(content=f"📝 Denoised (revision {state['step_count']})")]
    }


def final_report_node(state: TTDDRGraphState) -> dict:
    """Generate final comprehensive report."""
    model, _, _, _, _ = _get_components()
    
    research_text = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in state["search_history"]])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate comprehensive investor-grade feasibility study. Structure: Executive Summary, Site Context, Zoning, Environmental, Infrastructure, Market, Opportunities, Risks, Recommendations. Use professional language, cite sources, be thorough."),
        ("user", "Address: {address}\nBrief: {brief}\nResearch:\n{research}\nDraft:\n{draft}\n\nGenerate final report:")
    ])
    
    response = (prompt | model).invoke({
        "address": state["address"],
        "brief": state.get("brief", "General feasibility assessment"),
        "research": research_text[:3000],
        "draft": state["draft_report"][:1500]
    })
    
    return {
        "final_report": response.content,
        "messages": [AIMessage(content=f"✅ Final report: {len(response.content)} chars\n\n{response.content[:500]}...")]
    }


# ============================================================================
# Routing Logic
# ============================================================================

def should_continue_search(state: TTDDRGraphState) -> Literal["denoise", "final_report"]:
    """Decide whether to continue search or finalize."""
    if state["step_count"] >= 2 or (state["messages"] and "complete" in state["messages"][-1].content.lower()):
        return "final_report"
    return "denoise"


def after_denoise_routing(state: TTDDRGraphState) -> Literal["search", "final_report"]:
    """Decide whether to search more or finalize."""
    return "final_report" if state["step_count"] >= 2 else "search"


# ============================================================================
# Graph Construction
# ============================================================================

def create_graph():
    """Create TTD-DR agent graph.
    
    Flow: START → parse → plan → draft → search → denoise → (loop or finalize) → END
    """
    workflow = StateGraph(TTDDRGraphState)
    
    workflow.add_node("parse_input", parse_input_node)
    workflow.add_node("stage1_plan", stage1_plan_node)
    workflow.add_node("stage2_draft", stage2_draft_node)
    workflow.add_node("search", search_node)
    workflow.add_node("denoise", denoise_node)
    workflow.add_node("final_report", final_report_node)
    
    workflow.add_edge(START, "parse_input")
    workflow.add_edge("parse_input", "stage1_plan")
    workflow.add_edge("stage1_plan", "stage2_draft")
    workflow.add_edge("stage2_draft", "search")
    
    workflow.add_conditional_edges("search", should_continue_search, {"denoise": "denoise", "final_report": "final_report"})
    workflow.add_conditional_edges("denoise", after_denoise_routing, {"search": "search", "final_report": "final_report"})
    
    workflow.add_edge("final_report", END)
    
    return workflow.compile()


graph = create_graph()

