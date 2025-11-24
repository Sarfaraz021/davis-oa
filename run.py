#!/usr/bin/env python3
"""
Entry point for TTD-DR feasibility study report generation.

Usage:
    python run.py <address> [--brief "developer brief"]
    
Example:
    python run.py "123 Main St, San Francisco, CA" --brief "80-unit multifamily building"
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ttd_dr import generate_feasibility_report


def main():
    parser = argparse.ArgumentParser(
        description="Generate investor-grade feasibility study reports using TTD-DR"
    )
    parser.add_argument(
        "address",
        type=str,
        help="Parcel address to analyze"
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
        help="Output file path for the generated report"
    )
    
    args = parser.parse_args()
    
    print(f"Generating feasibility report for: {args.address}")
    if args.brief:
        print(f"Developer brief: {args.brief}")
    
    # Generate report
    report = generate_feasibility_report(
        address=args.address,
        brief=args.brief if args.brief else None
    )
    
    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    
    print(f"\nReport generated successfully: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

