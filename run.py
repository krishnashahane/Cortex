"""Cortex CLI."""
from __future__ import annotations

import argparse
import ipaddress
import sys


def _validate_iterations(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("iterations must be an integer") from exc
    if not 1 <= number <= 30:
        raise argparse.ArgumentTypeError("iterations must be between 1 and 30")
    return number


def _validate_host(value: str) -> str:
    if value == "localhost":
        return value
    try:
        if ipaddress.ip_address(value).is_loopback:
            return value
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(
        "dashboard is local-only; use localhost, 127.0.0.1, or ::1"
    )


def _research(args: argparse.Namespace) -> int:
    from cortex.engine import run_research

    print("Cortex research loop starting...")

    def on_event(item: dict) -> None:
        print(f"  [{item['agent']:<22}] {item['message']}")

    try:
        final = run_research(
            goal=args.goal,
            max_iterations=args.iters,
            on_event=on_event,
        )
    except Exception as exc:
        print(f"Research failed: {exc}", file=sys.stderr)
        return 1

    print()
    print("Done")
    print(f"Best score : {final.get('best_score', 0.0):.4f}")
    print(f"Stop reason: {final.get('termination_reason') or 'n/a'}")
    print(f"Report     : {final.get('report_path') or 'n/a'}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    print(f"Cortex dashboard -> http://{args.host}:{args.port}")
    uvicorn.run("cortex.app:app", host=args.host, port=args.port, reload=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cortex")
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="launch the local dashboard")
    serve.add_argument("--host", default="127.0.0.1", type=_validate_host)
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=_serve)

    research = sub.add_parser("research", help="run a headless research loop")
    research.add_argument("--goal", default="", type=str)
    research.add_argument("--iters", default=None, type=_validate_iterations)
    research.set_defaults(func=_research)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
