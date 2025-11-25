"""Main Entry Point for TTD-DR Feasibility Agent.

Usage:
    python run.py --address "123 Main St, San Francisco, CA" --brief "80-unit multifamily"
    python run.py --address "456 Oak Ave, Austin, TX"
"""

import argparse
import os
import json
from pathlib import Path
from dotenv import load_dotenv

from src.ttd_dr.agents.ttd_dr_agent import TTDDRAgent

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Generate feasibility study reports using TTD-DR")
    parser.add_argument("--address", type=str, required=True, help="Property address")
    parser.add_argument("--brief", type=str, default="", help="Developer brief")
    parser.add_argument("--output", type=str, default="reports/example_output.md", help="Output file path")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model")
    parser.add_argument("--max-steps", type=int, default=3, help="Maximum search steps (default: 3 for demo, increase for comprehensive research)")
    parser.add_argument("--no-evolution", action="store_true", help="Disable self-evolution")
    parser.add_argument("--no-diffusion", action="store_true", help="Disable diffusion refinement")
    args = parser.parse_args()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set")
        return
    if not os.getenv("TAVILY_API_KEY"):
        print("⚠️  TAVILY_API_KEY not set - web search may not work")
    
    print("=" * 80)
    print("TTD-DR: Test-Time Diffusion Deep Researcher")
    print("=" * 80)
    
    agent = TTDDRAgent(
        model_name=args.model,
        max_search_steps=args.max_steps,
        use_self_evolution=not args.no_evolution,
        use_diffusion=not args.no_diffusion
    )
    
    state = agent.run(args.address, args.brief)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(f"# Feasibility Study Report\n\n**Address:** {args.address}\n\n")
        if args.brief:
            f.write(f"**Brief:** {args.brief}\n\n")
        f.write(f"**Generated:** {state.metadata.get('timestamp', 'N/A')}\n")
        f.write(f"**Research Steps:** {len(state.search_history)}\n")
        f.write(f"**Revisions:** {state.revision_count}\n\n---\n\n{state.final_report}")
    
    print(f"\n✅ Report: {output_path}")
    
    state_path = output_path.parent / f"{output_path.stem}_state.json"
    with open(state_path, 'w') as f:
        json.dump(state.to_dict(), f, indent=2)
    
    print(f"📊 State: {state_path}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
