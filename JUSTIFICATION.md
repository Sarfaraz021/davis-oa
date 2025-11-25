## Design Choices & Justification

### Overview

This implementation realizes the **Test-Time Diffusion Deep Researcher (TTD-DR)** framework for generating investor-grade feasibility study reports. The system mirrors human research patterns through iterative planning, drafting, searching, and revision cycles.

---

### 1. State & History Representation

**Choice:** Centralized `AgentState` dataclass with structured search history

**Implementation:**
- `AgentState` maintains: query, plan, search history, draft versions, final report
- `SearchResult` objects store question-answer pairs with timestamps and sources
- Serializable to JSON for persistence and debugging

**Justification:**
- **Explicit state management** prevents "lost in the middle" issues common in long agent trajectories
- **Structured history** enables targeted context retrieval (last N results)
- **Immutable search results** create audit trail for transparency
- Inspired by [LangGraph state management](https://langchain-ai.github.io/langgraph/concepts/low_level/#state)

**Trade-offs:**
- Memory overhead for long research sessions (mitigated by context windowing)
- Serialization cost (acceptable for research use case)

---

### 2. Context Engineering & "Lost in the Middle"

**Choice:** Hierarchical context selection with recency bias

**Implementation:**
```python
def get_search_context(self, last_n: int = 5) -> str:
    recent = self.search_history[-last_n:]
    # Format and return
```

**Justification:**
- **Recency bias**: Recent findings most relevant for next search question ([Liu et al., 2023](https://arxiv.org/abs/2307.03172))
- **Truncation strategy**: Limit to 5 most recent Q&A pairs (~2K tokens)
- **Draft-guided search**: Current draft informs question generation, maintaining global coherence

**Alternative considered:**
- Semantic similarity-based retrieval: Rejected due to latency and complexity
- Full history: Causes context overflow and attention dilution

---

### 3. Retrieval & RAG Configuration

**Choice:** Hybrid retrieval with web search + vector database

**Implementation:**
- **Primary**: Tavily web search for real-time, location-specific data
- **Secondary**: Chroma vector DB for curated feasibility knowledge
- **RAG pattern**: Search → Retrieve → Synthesize answer (not raw documents)

**Justification:**
- **Web search essential**: Property data changes frequently, requires current information
- **Vector DB supplements**: Best practices, regulations, frameworks don't change often
- **Answer synthesis**: Reduces noise, prevents information overload in final report
- Follows [RAG best practices](https://www.anthropic.com/research/retrieval-augmented-generation) from Anthropic

**Configuration:**
- Web search: 5 results max, basic depth (speed vs. comprehensiveness)
- Vector DB: Top-3 semantic matches, 500 char truncation
- Lazy loading: Vector DB only initialized when first used (graceful degradation)

---

### 4. Diffusion Process Design

**Choice:** Draft-centric denoising with retrieval augmentation

**Implementation:**
1. Generate initial "noisy" draft from LLM internal knowledge
2. Each search step refines draft with new information
3. Draft guides next search question generation
4. Iterative loop until research complete

**Justification:**
- **Maintains global coherence**: Draft prevents fragmented section-by-section research ([Open Deep Research issue](https://github.com/langchain-ai/open_deep_research))
- **Timely integration**: Information incorporated immediately, not batched at end
- **Reduced information loss**: Draft serves as persistent memory across search steps
- Directly implements [Zhang et al., 2023](https://arxiv.org/abs/2304.04370) retrieval-augmented diffusion

**Denoising mechanism:**
```python
def _denoise_draft(state):
    # Incorporate last 3 search results into draft
    # Update sections, verify facts, add details
```

---

### 5. Self-Evolution Implementation

**Choice:** Component-wise optimization with LLM-as-judge

**Implementation:**
- Generate N variants of answers (N=2-3 for efficiency)
- Evaluate each with LLM-as-judge (Helpfulness + Completeness)
- Revise based on feedback
- Merge evolved variants

**Justification:**
- **Exploration vs. exploitation**: Multiple variants explore search space
- **Quality improvement**: Feedback loop raises answer quality ([Madaan et al., 2023](https://arxiv.org/abs/2303.17651))
- **Selective application**: Applied every 3 steps to balance cost/benefit
- Implements [Lee et al., 2025](https://arxiv.org/abs/2501.09891) self-evolution

**Hyperparameters:**
- Variants: 2-3 (diminishing returns beyond 3)
- Iterations: 1 (single feedback cycle sufficient)
- Frequency: Every 3rd step (cost-effective)

---

### 6. Stopping Criteria

**Choice:** Hybrid stopping with max steps + semantic completion

**Implementation:**
```python
# Demo mode (CLI default: 3, LangGraph: 2)
max_search_steps = 3  # Fast demos
# Production mode: 10-20 (configurable via --max-steps)
question = generate_next_question()
if question == "DONE":  # Semantic signal
    break
```

**Justification:**
- **Demo mode (3 steps)**: Fast demonstrations (~1-2 min), good for testing
- **Production mode (10-20 steps)**: Comprehensive research, aligns with paper benchmarks
- **Semantic completion**: LLM judges when plan adequately covered
- **Graceful degradation**: Always produces report, even if incomplete
- **Configurable**: Users can adjust based on needs vs. speed/cost trade-offs

**Alternative considered:**
- Plan coverage metric: Complex to implement reliably
- Fixed steps: Wastes compute on simple queries
- Cost-based: Unpredictable for users

**Configuration Rationale:**
- Default 3 steps for CLI enables quick testing without overwhelming demos
- LangGraph hardcoded to 2 steps for optimal UI experience
- Easy to scale up for production: `--max-steps 10` or `--max-steps 20`

---

### 7. Memory Bank Architecture

**Choice:** In-memory state with optional persistence

**Implementation:**
- Runtime: `AgentState` object in memory
- Persistence: JSON serialization to disk
- No database: Simplicity for single-run use case

**Justification:**
- **Simplicity**: No external dependencies (Redis, PostgreSQL)
- **Sufficient**: Feasibility studies are one-shot tasks, not conversational
- **Debuggable**: JSON state files enable post-hoc analysis

**Future enhancement:**
- Add checkpointing for long-running research (>1 hour)
- Implement LangGraph checkpointer for resume capability

---

### 8. Tool Integration

**Choice:** Minimal tool set (search only)

**Implementation:**
- Tavily web search: Primary information source
- Chroma retrieval: Supplementary knowledge
- No browsing, no code execution

**Justification:**
- **Scope management**: Focus on search-intensive tasks per paper
- **Reliability**: Fewer tools = fewer failure modes
- **Reproducibility**: Search results more stable than dynamic web pages

**Excluded tools:**
- Web browsing: Complex, slow, fragile
- Code execution: Not required for feasibility studies
- APIs: Property-specific APIs not universally available

---

### 9. Model Selection

**Choice:** GPT-4o-mini as default, configurable

**Justification:**
- **Cost-effective**: $0.15/1M input tokens vs. $2.50 for GPT-4
- **Sufficient quality**: Feasibility reports don't require frontier reasoning
- **Speed**: Faster inference enables more search iterations
- **Configurable**: Can upgrade to GPT-4 or Gemini for complex queries

---

### 10. Evaluation Strategy

**Choice:** LLM-as-judge with calibrated prompts

**Implementation:**
- Helpfulness: Intent, clarity, accuracy, language
- Comprehensiveness: Missing information assessment
- Scoring: 0-10 scale with textual feedback

**Justification:**
- **Scalability**: Human evaluation doesn't scale
- **Calibration**: Prompts designed to align with expert judgment
- **Actionable**: Textual feedback enables self-evolution
- Follows [Zheng et al., 2024](https://arxiv.org/abs/2306.05685) LLM-as-judge methodology

---

### 11. Error Handling & Robustness

**Choices:**
- Lazy vector DB loading (graceful degradation)
- Try-except around retrieval calls
- Default values for missing plan sections
- Truncation for oversized contexts

**Justification:**
- **Production readiness**: System works even with partial failures
- **User experience**: Always produces output, never crashes
- **Debugging**: Errors logged but don't halt execution

---

### 12. Performance Optimizations

**Implemented:**
- Context truncation (500 chars per document)
- Selective self-evolution (every 3rd step)
- Parallel variant generation (where possible)
- Early stopping on semantic completion

**Not implemented (future work):**
- Caching of search results
- Parallel search question generation
- Streaming output for long reports

---

### References

1. **TTD-DR Paper**: Han et al., 2025 - Deep Researcher with Test-Time Diffusion
2. **RAG**: Zhang et al., 2023 - Retrieval-Augmented Diffusion Models
3. **Self-Refine**: Madaan et al., 2023 - Self-Refine: Iterative Refinement with Self-Feedback
4. **Self-Evolution**: Lee et al., 2025 - Evolving Deeper LLM Thinking
5. **Lost in the Middle**: Liu et al., 2023 - Lost in the Middle: How Language Models Use Long Contexts
6. **LLM-as-Judge**: Zheng et al., 2024 - Judging LLM-as-a-Judge with MT-Bench
7. **LangGraph**: LangChain - [State Management Docs](https://langchain-ai.github.io/langgraph/)
8. **Anthropic RAG**: [RAG Best Practices](https://www.anthropic.com/research/retrieval-augmented-generation)

---

### Summary

This implementation prioritizes:
1. **Fidelity to paper**: Core TTD-DR algorithms implemented as described
2. **Production readiness**: Robust error handling, graceful degradation
3. **Debuggability**: Explicit state, serialization, logging
4. **Cost-effectiveness**: Efficient model usage, selective self-evolution
5. **Extensibility**: Modular design enables future enhancements

The system successfully generates investor-grade feasibility reports through iterative research, demonstrating the synergy between diffusion-style refinement and component-wise self-evolution.

