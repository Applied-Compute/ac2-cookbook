"""Build deterministic rating/length-stratified AC2 task files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lichess_puzzles.tool_contract import DEFAULT_TRIES

from .dataset import (
    SelectionConfig,
    iter_puzzle_rows,
    load_source_manifest,
    parse_bins,
    parse_split,
    select_tasks,
    write_selection,
)

DEFAULT_SOURCE = Path("data/raw/lichess_db_puzzle.csv.zst")
DEFAULT_OUTPUT = Path("data/prepared")
DEFAULT_RATING_BINS = "800-1199,1200-1599,1600-1999,2000-2399,2400-2799"
DEFAULT_LENGTH_BINS = "2,3,4,5-6"
DEFAULT_CALIBRATION_SPLIT = "calibration:0:1000:12"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--source-manifest", type=Path, default=Path("data/raw/source_manifest.json")
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-prefix", default="lichess-puzzles")
    parser.add_argument("--dataset-version", default="v1")
    parser.add_argument("--rating-bins", default=DEFAULT_RATING_BINS)
    parser.add_argument("--length-bins", default=DEFAULT_LENGTH_BINS)
    parser.add_argument(
        "--split",
        action="append",
        default=None,
        metavar="NAME:START:END:PER_STRATUM",
        help=(
            "Stable 0..9999 position-hash interval and target per rating/length cell. "
            f"Default: {DEFAULT_CALIBRATION_SPLIT}."
        ),
    )
    parser.add_argument("--seed", default="lichess-puzzles-v1")
    parser.add_argument("--min-popularity", type=int, default=70)
    parser.add_argument("--min-plays", type=int, default=100)
    parser.add_argument("--max-rating-deviation", type=int, default=100)
    parser.add_argument("--exclude-theme", action="append", default=["mateIn1"])
    parser.add_argument("--reasoning-budget", type=int)
    parser.add_argument(
        "--tries",
        type=int,
        default=DEFAULT_TRIES,
        help="Puzzle-wide budget of correctly formatted wrong moves.",
    )
    parser.add_argument("--oversample-factor", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_specs = args.split or [DEFAULT_CALIBRATION_SPLIT]
    config = SelectionConfig(
        rating_bins=parse_bins(args.rating_bins),
        length_bins=parse_bins(args.length_bins),
        splits=tuple(parse_split(item) for item in split_specs),
        seed=args.seed,
        min_popularity=args.min_popularity,
        min_plays=args.min_plays,
        max_rating_deviation=args.max_rating_deviation,
        excluded_themes=tuple(dict.fromkeys(args.exclude_theme)),
        reasoning_budget=args.reasoning_budget,
        tries=args.tries,
        oversample_factor=args.oversample_factor,
    )
    result = select_tasks(iter_puzzle_rows(args.source), config)
    source = load_source_manifest(args.source_manifest, archive=args.source)
    manifest_path = write_selection(
        result,
        output_dir=args.output_dir,
        dataset_prefix=args.dataset_prefix,
        dataset_version=args.dataset_version,
        config=config,
        source=source,
    )
    print(json.dumps({"manifest": str(manifest_path), **result.counts}, indent=2))


if __name__ == "__main__":
    main()
