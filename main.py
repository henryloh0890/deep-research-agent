import sys
import io

# Force UTF-8 encoding for Windows compatibility
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import argparse
from src.agent import run_research
from src.database import print_stats

def main():
    parser = argparse.ArgumentParser(
        description="Deep Research Agent — autonomous web research and report generation"
    )
    parser.add_argument(
        "topic",
        nargs="?",
        help="Research topic (e.g. 'future of renewable energy')"
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Show research history and stats"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress live progress output"
    )

    args = parser.parse_args()

    if args.history:
        print_stats()
        return

    if not args.topic:
        print("Usage: python main.py \"your research topic\"")
        print("       python main.py --history")
        sys.exit(1)

    run_research(topic=args.topic, verbose=not args.quiet)

if __name__ == "__main__":
    main()