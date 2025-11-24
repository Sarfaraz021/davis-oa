# TTD-DR: Test-Time Diffusion Deep Researcher

A production-ready implementation of the **Test-Time Diffusion Deep Researcher (TTD-DR)** framework for generating investor-grade feasibility study reports.

## Overview

This system implements the TTD-DR algorithm from the paper "Deep Researcher with Test-Time Diffusion" (Han et al., 2025), designed to generate comprehensive feasibility studies for real estate development projects.

### Key Features

- ✅ **Draft-Centric Research**: Maintains coherent draft throughout iterative search
- ✅ **Self-Evolution**: Component-wise optimization with LLM-as-judge
- ✅ **Diffusion Refinement**: Iterative denoising with retrieval augmentation
- ✅ **Hybrid Retrieval**: Web search (Tavily) + vector database (Chroma)
- ✅ **Graceful Degradation**: Works even without vector database
- ✅ **Production Ready**: Robust error handling, configurable parameters

---

## Setup

### 1. Clone & Install

```bash
git clone <repository-url>
cd davis-oa
pip install -r requirements.txt
```

### 2. Environment Variables

Copy the example environment file and add your API keys:

```bash
cp env.example .env
```

Edit `.env` and add:

```bash
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here  # Optional
```

### 3. Build Vector Database (Optional)

For enhanced knowledge retrieval:

```bash
jupyter notebook build_vector_db.ipynb
```

Run all cells to create the Chroma vector database from curated sources.

---

## Usage

### Basic Usage

```bash
python run.py --address "123 Main St, San Francisco, CA"
```

### With Developer Brief

```bash
python run.py \
  --address "456 Oak Ave, Austin, TX" \
  --brief "80-unit multifamily building"
```

### Advanced Options

```bash
python run.py \
  --address "789 Pine Rd, Seattle, WA" \
  --brief "Mixed-use development" \
  --output "reports/seattle_project.md" \
  --model "gpt-4o" \
  --max-steps 15 \
  --no-evolution  # Disable self-evolution
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--address` | Property address (required) | - |
| `--brief` | Developer brief | "" |
| `--output` | Output file path | `reports/example_output.md` |
| `--model` | OpenAI model | `gpt-4o-mini` |
| `--max-steps` | Max search/revision steps | 20 |
| `--no-evolution` | Disable self-evolution | False |
| `--no-diffusion` | Disable diffusion refinement | False |

---

## Architecture

### System Components

```
src/ttd_dr/
├── agents/
│   ├── graph.py              # LangGraph ReAct agent (baseline)
│   └── ttd_dr_agent.py       # Full TTD-DR implementation
├── planner/
│   └── planner.py            # Stage 1: Research plan generation
├── memory/
│   └── state.py              # State management & history
├── refinement/
│   ├── evaluator.py          # LLM-as-judge evaluators
│   └── self_evolution.py     # Self-evolution algorithm
├── retrieval/
│   └── retriever.py          # Chroma vector database
├── tools/
│   └── tools.py              # Tavily web search
└── prompts/
    └── prompt.py             # System prompts
```

### TTD-DR Pipeline

The agent follows a structured graph-based workflow:

```
                    START
                      ↓
              [parse_input]
                      ↓
            [Stage 1: Planning]
          Generate Research Plan
                      ↓
        [Stage 2a: Initial Draft]
      Generate Noisy Draft (Diffusion)
                      ↓
         ┌────────────┴────────────┐
         │  [Stage 2b: Search]     │
         │  • Generate Question    │
         │  • Web Search (Tavily)  │
         │  • KB Retrieval (Chroma)│
         │  • Synthesize Answer    │
         │  • Self-Evolution (3x)  │
         └────────────┬────────────┘
                      ↓
         ┌────────────┴────────────┐
         │ [Stage 2c: Denoise]     │
         │  • Refine Draft         │
         │  • Incorporate Research │
         │  • Improve Coherence    │
         └────────────┬────────────┘
                      ↓
              (Continue? 6 steps max)
                   /    \
                YES      NO
                 ↓        ↓
              Search   [Stage 3: Final Report]
                       Generate Comprehensive
                       Feasibility Study
                              ↓
                            END
```

**Key Stages:**
- **Stage 1**: Structured research plan generation
- **Stage 2a**: Initial "noisy" draft (diffusion start)
- **Stage 2b**: Iterative search with self-evolution (2 iterations)
- **Stage 2c**: Denoising (refine draft with new research)
- **Stage 3**: Final comprehensive report synthesis

**Note**: The LangGraph version runs 2 search-denoise iterations for faster demos. The CLI version (`run.py`) supports up to 20 iterations for comprehensive research.

---

## Example Output

See `reports/example_output.md` for a complete generated feasibility study report.

### Report Structure

1. **Executive Summary**
2. **Site Context & Location**
3. **Zoning & Regulatory Analysis**
4. **Environmental Constraints**
5. **Infrastructure & Utilities**
6. **Market Analysis**
7. **Development Opportunities**
8. **Risks & Challenges**
9. **Recommendations**

---

## Configuration

### Model Selection

- **gpt-4o-mini** (default): Cost-effective, fast, sufficient quality
- **gpt-4o**: Higher quality for complex queries
- **gpt-4**: Maximum reasoning capability

### Search Steps

- **10 steps**: Quick preliminary analysis
- **20 steps** (default): Comprehensive research
- **30+ steps**: Deep dive for complex projects

### Algorithms

- **Self-Evolution**: Improves answer quality through variants and feedback
- **Diffusion**: Maintains coherent draft throughout research
- Both can be disabled for faster, simpler operation

---

## LangGraph Studio

The system includes a LangGraph-compatible agent for visual debugging:

```bash
langgraph dev
```

Access Studio UI at: `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`

---

## Design Decisions

See `JUSTIFICATION.md` for detailed explanations of:
- State & history representation
- Context engineering strategies
- RAG configuration choices
- Diffusion process design
- Self-evolution implementation
- Stopping criteria
- Error handling approaches

---

## Performance

### Typical Execution

- **Time**: 3-5 minutes for 20 search steps
- **Cost**: ~$0.50-1.00 per report (gpt-4o-mini)
- **Quality**: Investor-grade, comprehensive analysis

### Optimization Tips

1. Use `gpt-4o-mini` for cost savings
2. Reduce `--max-steps` for faster results
3. Disable `--no-evolution` for speed
4. Build vector DB for better knowledge retrieval

---

## Troubleshooting

### "Vector database not found"

The system works without it! Web search provides real-time data. To enable:
```bash
jupyter notebook build_vector_db.ipynb
```

### "TAVILY_API_KEY not set"

Get a free API key at: https://tavily.com
Add to `.env` file.

### "Rate limit exceeded"

Reduce `--max-steps` or wait for rate limit reset.

### Import errors

```bash
pip install -r requirements.txt
```

---

## Testing

Run the baseline agent:
```bash
python -c "from src.ttd_dr.agents.graph import create_agent; agent = create_agent(); print('✅ Agent loaded')"
```

Test TTD-DR:
```bash
python run.py --address "Test Address" --max-steps 3
```

---

## Contributing

This is a research implementation. For production use:
1. Add caching for search results
2. Implement streaming output
3. Add progress callbacks
4. Enhance error recovery
5. Add unit tests

---

## References

- **Paper**: Han et al., 2025 - "Deep Researcher with Test-Time Diffusion"
- **LangGraph**: https://langchain-ai.github.io/langgraph/
- **Tavily**: https://tavily.com
- **Chroma**: https://www.trychroma.com

---

## License

See LICENSE file for details.

---

## Citation

```bibtex
@article{han2025ttddr,
  title={Deep Researcher with Test-Time Diffusion},
  author={Han, Rujun and Chen, Yanfei and others},
  journal={arXiv preprint arXiv:2507.16075},
  year={2025}
}
```

---

**Built with ❤️ for real estate feasibility analysis**

