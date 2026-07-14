"""Command-line interface for vocalcoverage."""

from __future__ import annotations

import argparse
import json
import sys

from .pipeline import analyze


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vocalcoverage")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze vocal coverage from a mix and its vocal stem"
    )
    analyze_parser.add_argument("mix", help="Path to the full mix audio file")
    analyze_parser.add_argument("vocals", help="Path to the separated vocal stem audio file")
    analyze_parser.add_argument("--frame-duration", type=float, default=1.0)
    analyze_parser.add_argument("--f0-min", type=float, default=80.0)
    analyze_parser.add_argument("--f0-max", type=float, default=1000.0)
    analyze_parser.add_argument("--silence-threshold-db", type=float, default=-40.0)
    analyze_parser.add_argument("--ratio-threshold", type=float, default=0.1)
    analyze_parser.add_argument("--f0-confidence-threshold", type=float, default=0.2)
    analyze_parser.add_argument("--smoothing-window", type=int, default=3)
    analyze_parser.add_argument("--json", action="store_true", help="Output raw JSON")

    return parser


def _print_human(result: dict) -> None:
    coverage_pct = result["vocal_coverage"] * 100
    print(f"Vocal coverage: {coverage_pct:.1f}%")
    print(f"Frames analyzed: {result['frame_count']} (silent: {result['silent_frame_count']})")
    print(f"Segments: {len(result['segments'])}")
    for segment in result["segments"]:
        ratio_mean = segment["ratio_mean"]
        f0_confidence = segment["f0_confidence"]
        print(
            f"  [{segment['start']:.1f}s - {segment['end']:.1f}s] "
            f"ratio={ratio_mean:.2f} f0_confidence={f0_confidence:.2f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze":
        try:
            result = analyze(
                mix_path=args.mix,
                vocals_path=args.vocals,
                frame_duration=args.frame_duration,
                f0_min=args.f0_min,
                f0_max=args.f0_max,
                silence_threshold_db=args.silence_threshold_db,
                ratio_threshold=args.ratio_threshold,
                f0_confidence_threshold=args.f0_confidence_threshold,
                smoothing_window=args.smoothing_window,
            )
        except (RuntimeError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            _print_human(result)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
