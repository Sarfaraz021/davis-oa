"""
Main Entry Point for TTD-DR Feasibility Agent

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
    parser = argparse.ArgumentParser(
        description="Generate feasibility study reports using TTD-DR"
    )
    parser.add_argument(
        "--address",
        type=str,
        required=True,
        help="Property address for feasibility study"
    )
    parser.add_argument(
        "--brief",
        type=str,
        default="",
        help="Optional developer brief (e.g., '80-unit multifamily building')"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/example_output.md",
        help="Output file path for the report"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Maximum search/revision steps"
    )
    parser.add_argument(
        "--no-evolution",
        action="store_true",
        help="Disable self-evolution algorithm"
    )
    parser.add_argument(
        "--no-diffusion",
        action="store_true",
        help="Disable diffusion refinement"
    )
    
    args = parser.parse_args()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not set in environment")
        return
    
    if not os.getenv("TAVILY_API_KEY"):
        print("⚠️  Warning: TAVILY_API_KEY not set - web search may not work")
    
    print("=" * 80)
    print("TTD-DR: Test-Time Diffusion Deep Researcher")
    print("Feasibility Study Report Generation")
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
        f.write(f"# Feasibility Study Report\n\n")
        f.write(f"**Address:** {args.address}\n\n")
        if args.brief:
            f.write(f"**Developer Brief:** {args.brief}\n\n")
        f.write(f"**Generated:** {state.metadata.get('timestamp', 'N/A')}\n\n")
        f.write(f"**Research Steps:** {len(state.search_history)}\n\n")
        f.write(f"**Revisions:** {state.revision_count}\n\n")
        f.write("---\n\n")
        f.write(state.final_report)
    
    print(f"\n✅ Report saved to: {output_path}")
    
    state_path = output_path.parent / f"{output_path.stem}_state.json"
    with open(state_path, 'w') as f:
        json.dump(state.to_dict(), f, indent=2)
    
    print(f"📊 State saved to: {state_path}")
    print("\n" + "=" * 80)
    print("Report Generation Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
