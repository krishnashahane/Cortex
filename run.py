"""Cortex CLI.

Usage:
    python run.py serve [--host H] [--port P]   # launch dashboard on localhost
    python run.py research [--goal ...] [--iters N]   # run headless in terminal
"""
from __future__ import annotations

import argparse

from cortex.engine import run_research


def _research(args: argparse.Namespace) -> None:
    print("◈ Cortex research loop starting...\n")

    def on_event(ev: dict) -> None:
        print(f"  [{ev['agent']:<20}] {ev['message']}")

    final = run_research(goal=args.goal, max_iterations=args.iters, on_event=on_event)
    print("\n── Done ──")
    print(f"Best score : {final.get('best_score', 0):.4f}")
    print(f"Stop reason: {final.get('termination_reason')}")
    print(f"Report     : {final.get('report_path')}")


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    print(f"◈ Cortex dashboard → http://{args.host}:{args.port}")
    uvicorn.run("server.app:app", host=args.host, port=args.port, reload=False)


def main() -> None:
    p = argparse.ArgumentParser(prog="cortex")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="launch the web dashboard")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=_serve)

    r = sub.add_parser("research", help="run a headless research loop")
    r.add_argument("--goal", default="")
    r.add_argument("--iters", type=int, default=None)
    r.set_defaults(func=_research)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
