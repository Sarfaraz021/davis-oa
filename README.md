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

### Quick Start

```bash
# 1. Clone repository
git clone https://github.com/Sarfaraz021/davis-oa.git
git checkout feat/ttd-dr-system
cd davis-oa

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
cp env.example .env
# Edit .env and add your OPENAI_API_KEY and TAVILY_API_KEY

# 4. Run via CLI
python run.py --address "123 Main St, San Francisco, CA" --brief "80-unit multifamily"

# OR run via LangGraph Studio
langgraph dev
```

---

## Setup

### 1. Clone and switch the feat branch & Install

```bash
git clone https://github.com/Sarfaraz021/davis-oa.git
git checkout feat/ttd-dr-system
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
LANGSMITH_API_KEY=your_langsmith_api_key_here  # Optional if you want to runo it on langggraph/langsmith ui
```

### 3. Build Vector Database (Optional) if you want to add more data in chomra vector store, currently it contains the content from these urls "/davis-oa/data/sources.yaml" just for demo purpose 

For enhanced knowledge retrieval:

```bash
jupyter notebook build_vector_db.ipynb
```

Run all cells to create the Chroma vector database from curated sources.

---

## Usage

You can run the TTD-DR agent in two ways:

### Option 1: Command Line Interface (CLI)

**Basic Usage:**
```bash
python run.py --address "123 Main St, San Francisco, CA"
```

**With Developer Brief:**
```bash
python run.py \
  --address "456 Oak Ave, Austin, TX" \
  --brief "80-unit multifamily building"
```

**Advanced Options:**
```bash
python run.py \
  --address "789 Pine Rd, Seattle, WA" \
  --brief "Mixed-use development" \
  --output "reports/seattle_project.md" \
  --model "gpt-4o" \
  --max-steps 20 \
  --no-evolution  # Disable self-evolution
```

**Command-Line Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--address` | Property address (required) | - |
| `--brief` | Developer brief | "" |
| `--output` | Output file path | `reports/example_output.md` |
| `--model` | OpenAI model | `gpt-4o-mini` |
| `--max-steps` | Max search/revision steps | `3` (demo), increase to 10-20 for comprehensive |
| `--no-evolution` | Disable self-evolution | False |
| `--no-diffusion` | Disable diffusion refinement | False |

**Note:** Default is set to 3 steps for fast demos. For comprehensive feasibility studies, use `--max-steps 10` (thorough) or `--max-steps 20` (very comprehensive).

---

### Option 2: LangGraph Studio (Visual Interface)

**Start the LangGraph development server:**

```bash
langgraph dev
```

This will:
- Start the LangGraph API server on `http://127.0.0.1:2024`
- Automatically open LangGraph Studio in your browser
- Enable visual debugging and monitoring of the agent workflow

**Using LangGraph Studio:**

1. **Access Studio:** The browser will open automatically to:
   ```
   https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
   ```

2. **Select Agent:** Choose `ttd_dr_agent` from the graph selector

3. **Input:** In the input panel, provide input to run in message field

4. **Run:** Click the "Run" button to start the agent

5. **Monitor:** Watch the agent progress through:
   - Stage 1: Planning
   - Stage 2: Initial Draft
   - Stage 2b: Iterative Search (2 iterations for demo)
   - Stage 2c: Denoising
   - Stage 3: Final Report

**Benefits of LangGraph Studio:**
- 📊 Visual graph execution flow
- 🔍 Step-by-step debugging
- 📝 Inspect state at each node
- 🎯 See intermediate outputs
- ⚡ Real-time progress monitoring

**CLI vs LangGraph Studio Comparison:**

| Feature | CLI (`run.py`) | LangGraph Studio |
|---------|----------------|------------------|
| **Best For** | Production runs | Development & debugging |
| **Search Steps (Default)** | 3 (demo) | 2 (demo) |
| **Search Steps (Max)** | Configurable (10-20+ recommended) | 2 (hardcoded for UI speed) |
| **Output** | Markdown file + JSON state | Visual graph + messages |
| **Visibility** | Terminal logs | Full graph visualization |
| **Speed** | Configurable (3 steps ~1-2min) | Fast (2 steps ~1min) |
| **Use Case** | Final reports & production | Testing & debugging |

**Demo Configuration Note:**
- Both CLI and LangGraph use minimal iterations (2-3) by default for fast demonstrations
- For comprehensive research, increase CLI steps: `--max-steps 10` (thorough) or `--max-steps 20` (very comprehensive)
- LangGraph hardcoded to 2 steps for optimal UI experience; modify `ttd_dr_agent.py` line 305 to increase

---

## Architecture

### System Components

The codebase has been optimized and cleaned (~50% reduction in lines) while maintaining full functionality:

```
src/ttd_dr/
├── agents/
│   └── ttd_dr_agent.py       # Full TTD-DR implementation (441 lines, optimized)
├── planner/
│   └── planner.py            # Research plan generation (28 lines)
├── memory/
│   └── state.py              # State management (44 lines)
├── refinement/
│   ├── evaluator.py          # LLM-as-judge (39 lines)
│   └── self_evolution.py     # Self-evolution algorithm (45 lines)
├── retrieval/
│   └── retriever.py          # Chroma vector database (50 lines)
├── tools/
│   └── tools.py              # Tavily web search (7 lines)
└── prompts/
    └── prompt.py             # System prompts (13 lines)
```

**Code Quality:**
- ✅ Clean, concise, maintainable
- ✅ No linting errors
- ✅ Consistent style throughout
- ✅ Optimized for readability and performance

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
- **Stage 2b**: Iterative search with self-evolution (configurable: 2-3 for demos, 10-20 for production)
- **Stage 2c**: Denoising (refine draft with new research)
- **Stage 3**: Final comprehensive report synthesis

**Demo Configuration:**
- **LangGraph**: Hardcoded to 2 iterations for fast UI demonstrations
- **CLI**: Default 3 iterations (demo mode), configurable via `--max-steps` flag
- **Production**: Recommended 10-20 iterations for comprehensive feasibility studies

---

## Example Output

**Note:** To generate an example report, run:
```bash
# Quick demo (3 steps, ~1-2 min)
python run.py --address "123 Main St, San Francisco, CA" --brief "80-unit multifamily" --output reports/example_output.md

# Thorough analysis (10 steps, ~3-4 min)
python run.py --address "123 Main St, San Francisco, CA" --brief "80-unit multifamily" --output reports/example_output.md --max-steps 10
```

The generated report will be saved to `reports/example_output.md`.

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

- **3 steps** (default): Fast demo mode (~1-2 minutes)
- **10 steps**: Thorough preliminary analysis (~3-4 minutes)
- **20 steps**: Comprehensive research (~5-8 minutes)
- **30+ steps**: Deep dive for complex projects (~10+ minutes)

### Algorithms

- **Self-Evolution**: Improves answer quality through variants and feedback
- **Diffusion**: Maintains coherent draft throughout research
- Both can be disabled for faster, simpler operation

---

## Configuration Files

### langgraph.json

The `langgraph.json` file configures the LangGraph deployment:

```json
{
  "graphs": {
    "ttd_dr_agent": "./src/ttd_dr/agents/ttd_dr_agent.py:graph"
  },
  "env": ".env",
  "python_version": "3.11"
}
```

This tells LangGraph Studio where to find the compiled graph for visual debugging.

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

- **Demo Mode (3 steps)**: ~1-2 minutes, ~$0.10-0.20 per report
- **Thorough (10 steps)**: ~3-4 minutes, ~$0.30-0.50 per report
- **Comprehensive (20 steps)**: ~5-8 minutes, ~$0.50-1.00 per report
- **Cost**: Using gpt-4o-mini (~$0.15/1M tokens)
- **Quality**: Scales with steps - more steps = more comprehensive

### Optimization Tips

1. Use `gpt-4o-mini` for cost savings (default)
2. Start with 3 steps for quick testing
3. Use `--max-steps 10` for production reports
4. Disable `--no-evolution` for additional speed
5. Build vector DB for better knowledge retrieval

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

### LangGraph Studio not opening

1. Check the server is running: `http://127.0.0.1:2024/docs`
2. Manually open: `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`
3. Verify `.env` file has required API keys
4. Check terminal for error messages

### LangGraph: "Graph not found"

Ensure `langgraph.json` points to the correct graph:
```json
{
  "graphs": {
    "ttd_dr_agent": "./src/ttd_dr/agents/ttd_dr_agent.py:graph"
  }
}
```

---

## Testing

Quick smoke test (uses default 3 steps):
```bash
python run.py --address "123 Main St, San Francisco, CA"
```

Test with custom steps:
```bash
# Fast test (3 steps, ~1-2 min)
python run.py --address "Test Address" --max-steps 3

# Production test (10 steps, ~3-4 min)
python run.py --address "Test Address" --max-steps 10
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

**Built by Ahmed for real estate feasibility analysis**

