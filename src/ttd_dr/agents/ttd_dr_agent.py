"""
TTD-DR Agent - Test-Time Diffusion Deep Researcher
Implements the complete TTD-DR algorithm from the paper.
"""

import os
from typing import Optional
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


def create_graph():
    """Create LangGraph-compatible graph with full TTD-DR stages."""
    from typing import TypedDict, Annotated, Sequence, Dict, Any
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
    
    class TTDDRGraphState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], "Conversation messages"]
        address: str
        brief: str
        plan: Dict[str, Any]
        search_history: list
        draft_report: str
        final_report: str
        step_count: int
    
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    planner = ResearchPlanner(model)
    evaluator = LLMEvaluator(model)
    self_evolution = SelfEvolution(model, evaluator)
    
    try:
        retriever = ChromaRetriever()
    except FileNotFoundError:
        retriever = None
    
    def parse_input(state: TTDDRGraphState) -> TTDDRGraphState:
        """Parse user input."""
        messages = state["messages"]
        last_msg = messages[-1]
        
        if isinstance(last_msg, dict):
            user_input = last_msg.get("content", "")
        else:
            user_input = last_msg.content
        
        state["address"] = user_input
        state["brief"] = ""
        state["search_history"] = []
        state["step_count"] = 0
        
        return state
    
    def stage1_plan(state: TTDDRGraphState) -> TTDDRGraphState:
        """Stage 1: Generate research plan."""
        plan = planner.generate_plan(state["address"], state["brief"])
        state["plan"] = plan
        
        msg = AIMessage(content=f"📝 Stage 1: Generated plan with {len(plan.get('sections', []))} sections")
        state["messages"].append(msg)
        
        return state
    
    def stage2_initial_draft(state: TTDDRGraphState) -> TTDDRGraphState:
        """Generate initial noisy draft (diffusion start)."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate initial draft feasibility study from internal knowledge."),
            ("user", "Address: {address}\nGenerate initial draft.")
        ])
        
        chain = prompt | model
        response = chain.invoke({"address": state["address"]})
        
        state["draft_report"] = response.content
        state["messages"].append(AIMessage(content="✍️ Diffusion: Initial draft generated"))
        
        return state
    
    def stage2_search_question(state: TTDDRGraphState) -> TTDDRGraphState:
        """Generate search question."""
        if state["step_count"] >= 8:
            state["messages"].append(AIMessage(content="✓ Research complete (max steps)"))
            return state
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate next search question or 'DONE'."),
            ("user", "Address: {address}\nStep: {step}\nPrevious: {history}")
        ])
        
        history = "\n".join([f"Q: {q}" for q, _ in state["search_history"][-3:]])
        
        chain = prompt | model
        response = chain.invoke({
            "address": state["address"],
            "step": state["step_count"] + 1,
            "history": history or "None"
        })
        
        question = response.content.strip()
        if "DONE" in question.upper():
            return state
        
        state["messages"].append(AIMessage(content=f"🔍 Q{state['step_count']+1}: {question[:60]}..."))
        
        web_results = web_search_tool.invoke({"query": question})
        
        kb_results = ""
        if retriever:
            try:
                kb_docs = retriever.retrieve(question, top_k=2)
                kb_results = "\n".join([d['content'][:200] for d in kb_docs])
            except:
                pass
        
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", "Synthesize answer from search results."),
            ("user", "Q: {q}\nWeb: {web}\nKB: {kb}")
        ])
        
        chain = answer_prompt | model
        answer_resp = chain.invoke({
            "q": question,
            "web": str(web_results)[:1500],
            "kb": kb_results[:500] if kb_results else "N/A"
        })
        
        answer = answer_resp.content
        
        if state["step_count"] % 3 == 0:
            state["messages"].append(AIMessage(content="🧬 Self-evolution: Improving answer..."))
            answer = self_evolution.evolve_answer(question, answer, num_variants=2)
        
        state["search_history"].append((question, answer))
        state["step_count"] += 1
        
        return state
    
    def stage2_denoise(state: TTDDRGraphState) -> TTDDRGraphState:
        """Denoise draft with new information."""
        if not state["search_history"]:
            return state
        
        latest = state["search_history"][-2:]
        latest_text = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in latest])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Refine draft with new research."),
            ("user", "Draft: {draft}\n\nNew: {new}")
        ])
        
        chain = prompt | model
        response = chain.invoke({
            "draft": state["draft_report"][:800],
            "new": latest_text
        })
        
        state["draft_report"] = response.content
        state["messages"].append(AIMessage(content=f"📝 Denoising: Draft updated (rev {state['step_count']})"))
        
        return state
    
    def stage3_final_report(state: TTDDRGraphState) -> TTDDRGraphState:
        """Stage 3: Generate final report."""
        research = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in state["search_history"]])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Generate comprehensive feasibility study:
1. Executive Summary
2. Site Context
3. Zoning Analysis
4. Environmental Constraints
5. Infrastructure
6. Market Analysis
7. Opportunities
8. Risks
9. Recommendations"""),
            ("user", "Address: {addr}\nResearch: {research}\nDraft: {draft}")
        ])
        
        chain = prompt | model
        response = chain.invoke({
            "addr": state["address"],
            "research": research[:2000],
            "draft": state["draft_report"][:1000]
        })
        
        state["final_report"] = response.content
        state["messages"].append(AIMessage(content=f"✅ Stage 3: Final report ({len(response.content)} chars)"))
        
        return state
    
    def should_continue_search(state: TTDDRGraphState) -> str:
        """Decide if more search needed."""
        if state["step_count"] >= 8:
            return "denoise_final"
        if state["search_history"] and state["step_count"] % 2 == 0:
            return "denoise"
        return "search"
    
    def route_after_denoise(state: TTDDRGraphState) -> str:
        """Route after denoising."""
        if state["step_count"] >= 8:
            return "final_report"
        return "search"
    
    workflow = StateGraph(TTDDRGraphState)
    
    workflow.add_node("parse_input", parse_input)
    workflow.add_node("stage1_plan", stage1_plan)
    workflow.add_node("stage2_draft", stage2_initial_draft)
    workflow.add_node("search", stage2_search_question)
    workflow.add_node("denoise", stage2_denoise)
    workflow.add_node("denoise_final", stage2_denoise)
    workflow.add_node("final_report", stage3_final_report)
    
    workflow.set_entry_point("parse_input")
    workflow.add_edge("parse_input", "stage1_plan")
    workflow.add_edge("stage1_plan", "stage2_draft")
    workflow.add_edge("stage2_draft", "search")
    
    workflow.add_conditional_edges(
        "search",
        should_continue_search,
        {
            "search": "search",
            "denoise": "denoise",
            "denoise_final": "denoise_final"
        }
    )
    
    workflow.add_conditional_edges(
        "denoise",
        route_after_denoise,
        {
            "search": "search",
            "final_report": "final_report"
        }
    )
    
    workflow.add_edge("denoise_final", "final_report")
    workflow.add_edge("final_report", END)
    
    return workflow.compile()


graph = create_graph()

