# Quick Start Guide

## 🚀 Get Started in 2 Minutes

### 1. Setup (30 seconds)

```bash
# Clone and checkout branch
git clone https://github.com/Sarfaraz021/davis-oa.git
git checkout feat/ttd-dr-system
cd davis-oa

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp env.example .env
# Edit .env and add OPENAI_API_KEY and TAVILY_API_KEY
```

### 2. Run Your First Report (1-2 minutes)

```bash
# Quick demo with default 3 steps
python run.py --address "123 Main St, San Francisco, CA" --brief "80-unit multifamily"
```

**Output:** `reports/example_output.md` (~1-2 minutes)

---

## 🎨 Visual Debugging with LangGraph

```bash
# Start LangGraph Studio
langgraph dev

# Browser opens automatically to:
# https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

# In Studio:
# 1. Select "ttd_dr_agent"
# 2. Enter address in message field: "123 Main St, San Francisco, CA"
# 3. Click "Run"
# 4. Watch the visual graph execution!
```

---

## ⚙️ Configuration Options

### Demo Mode (Default - Fast)
```bash
# 3 steps, ~1-2 minutes, $0.10-0.20
python run.py --address "Your Address"
```

### Thorough Analysis
```bash
# 10 steps, ~3-4 minutes, $0.30-0.50
python run.py --address "Your Address" --max-steps 10
```

### Comprehensive Research
```bash
# 20 steps, ~5-8 minutes, $0.50-1.00
python run.py --address "Your Address" --max-steps 20
```

---

## 📊 What Gets Generated

### Report Sections (Investor-Grade)
1. ✅ Executive Summary
2. ✅ Site Context & Location
3. ✅ Zoning & Regulatory Analysis
4. ✅ Environmental Constraints
5. ✅ Infrastructure & Utilities
6. ✅ Market Analysis
7. ✅ Development Opportunities
8. ✅ Risks & Challenges
9. ✅ Recommendations

### State Files
- `reports/example_output.md` - Final report
- `reports/example_output_state.json` - Complete agent state

---

## 🔧 Troubleshooting

### "OPENAI_API_KEY not set"
```bash
# Add to .env file
OPENAI_API_KEY=your_key_here
```

### "TAVILY_API_KEY not set"
```bash
# Get free key at https://tavily.com
# Add to .env file
TAVILY_API_KEY=your_key_here
```

### "Vector database not found"
```bash
# Optional - system works without it!
# To enable, run:
jupyter notebook build_vector_db.ipynb
```

---

## 🎯 TTD-DR Features

- ✅ **Self-Evolution**: Answer quality improvement through variants & LLM-as-judge
- ✅ **Diffusion Refinement**: Iterative draft denoising with research integration
- ✅ **Hybrid Retrieval**: Web search (Tavily) + Vector DB (Chroma)
- ✅ **Structured Planning**: Automatic research plan generation
- ✅ **Configurable**: Adjust steps, model, algorithms on/off

---

## 📖 Full Documentation

- **README.md** - Complete setup and usage guide
- **JUSTIFICATION.md** - Design choices and trade-offs
- **langgraph.json** - LangGraph configuration

---

## ⏱️ Time & Cost Estimates

| Mode | Steps | Time | Cost (GPT-4o-mini) | Use Case |
|------|-------|------|-------------------|----------|
| **Demo** | 3 | 1-2 min | $0.10-0.20 | Testing, demos |
| **Thorough** | 10 | 3-4 min | $0.30-0.50 | Production reports |
| **Comprehensive** | 20 | 5-8 min | $0.50-1.00 | Complex projects |

---

## 🎓 Understanding the Process

### Stage 1: Planning
Generate structured research plan with sections and questions

### Stage 2a: Initial Draft
Create "noisy" draft from LLM's internal knowledge

### Stage 2b: Iterative Search
- Generate focused search question
- Search web (Tavily) and knowledge base (Chroma)
- Apply self-evolution (every 3rd step)
- Synthesize answer

### Stage 2c: Denoising
Refine draft by incorporating latest research findings

### Stage 3: Final Report
Generate comprehensive investor-grade feasibility study

---

## 🚀 Ready to Submit?

1. ✅ Generate example report: `python run.py --address "123 Main St, San Francisco, CA" --output reports/example_output.md`
2. ✅ Test LangGraph: `langgraph dev`
3. ✅ Verify output: `ls -la reports/example_output.md`
4. ✅ Commit and push!

---

**Need help?** Check README.md for detailed documentation.

**Built by Ahmed for real estate feasibility analysis**

