"""Metadata-preserving, deterministic stratified sampling from Lichess CSV rows."""

from __future__ import annotations

import csv
import hashlib
import heapq
import io
import itertools
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO

import zstandard
from ac2.runtime import Message, Task

from lichess_puzzles.chess_logic import OBSERVATION_FORMAT, position_prompt, validate_puzzle
from lichess_puzzles.tool_contract import DEFAULT_TRIES

LICHESS_SOURCE_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
LICHESS_LICENSE = "CC0-1.0"
EXPECTED_COLUMNS = (
    "PuzzleId",
    "FEN",
    "Moves",
    "Rating",
    "RatingDeviation",
    "Popularity",
    "NbPlays",
    "Themes",
    "GameUrl",
    "OpeningTags",
    "DailyDate",
)


@dataclass(frozen=True)
class PuzzleRow:
    """All fields from one current Lichess puzzle CSV row."""

    puzzle_id: str
    source_fen: str
    moves: tuple[str, ...]
    rating: int
    rating_deviation: int
    popularity: int
    nb_plays: int
    themes: tuple[str, ...]
    game_url: str
    opening_tags: tuple[str, ...]
    daily_date: int | None

    @classmethod
    def from_csv(cls, row: dict[str, str | None]) -> PuzzleRow:
        def required(name: str) -> str:
            value = row.get(name)
            if value is None or not value.strip():
                raise ValueError(f"missing required Lichess field {name}")
            return value.strip()

        daily = (row.get("DailyDate") or "").strip()
        return cls(
            puzzle_id=required("PuzzleId"),
            source_fen=required("FEN"),
            moves=tuple(required("Moves").split()),
            rating=int(required("Rating")),
            rating_deviation=int(required("RatingDeviation")),
            popularity=int(required("Popularity")),
            nb_plays=int(required("NbPlays")),
            themes=tuple(filter(None, (row.get("Themes") or "").split())),
            game_url=(row.get("GameUrl") or "").strip(),
            opening_tags=tuple(filter(None, (row.get("OpeningTags") or "").split())),
            daily_date=int(daily) if daily else None,
        )


@dataclass(frozen=True)
class IntBin:
    """Inclusive integer interval used as a sampling stratum."""

    low: int
    high: int

    def __post_init__(self) -> None:
        if self.low < 0 or self.high < self.low:
            raise ValueError(f"invalid inclusive bin {self.low}-{self.high}")

    @property
    def label(self) -> str:
        return str(self.low) if self.low == self.high else f"{self.low}-{self.high}"

    def contains(self, value: int) -> bool:
        return self.low <= value <= self.high


@dataclass(frozen=True)
class SplitSpec:
    """A stable split interval and target count for every rating/length cell."""

    name: str
    bucket_start: int
    bucket_end: int
    per_stratum: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("split name cannot be empty")
        if not 0 <= self.bucket_start < self.bucket_end <= 10_000:
            raise ValueError("split buckets must satisfy 0 <= start < end <= 10000")
        if self.per_stratum <= 0:
            raise ValueError("per_stratum must be positive")

    @property
    def fraction(self) -> float:
        return (self.bucket_end - self.bucket_start) / 10_000


@dataclass(frozen=True)
class SelectionConfig:
    """Complete deterministic sampling policy."""

    rating_bins: tuple[IntBin, ...]
    length_bins: tuple[IntBin, ...]
    splits: tuple[SplitSpec, ...]
    seed: str = "lichess-puzzles-v1"
    min_popularity: int = 70
    min_plays: int = 100
    max_rating_deviation: int = 100
    excluded_themes: tuple[str, ...] = ("mateIn1",)
    reasoning_budget: int | None = None
    tries: int = DEFAULT_TRIES
    oversample_factor: int = 4

    def __post_init__(self) -> None:
        if not self.rating_bins or not self.length_bins or not self.splits:
            raise ValueError("rating bins, length bins, and splits cannot be empty")
        if self.oversample_factor < 2:
            raise ValueError("oversample_factor must be at least 2")
        if isinstance(self.tries, bool) or not isinstance(self.tries, int) or self.tries <= 0:
            raise ValueError("tries must be a positive integer")
        for left, right in itertools.pairwise(self.splits):
            if left.bucket_end > right.bucket_start:
                raise ValueError(f"split intervals overlap: {left.name} and {right.name}")


@dataclass(frozen=True)
class SelectedTask:
    """Task plus deterministic ordering and stratum metadata."""

    task: Task
    puzzle_id: str
    sample_rank: int
    rating_bin: str
    length_bin: str


@dataclass(frozen=True)
class SelectionResult:
    """Selected tasks and auditable source/filter counts."""

    tasks: dict[str, list[Task]]
    counts: dict[str, int]
    strata: dict[str, dict[str, int]]


def parse_bins(spec: str) -> tuple[IntBin, ...]:
    """Parse comma-separated inclusive bins such as ``800-1199,1200-1599``."""

    bins: list[IntBin] = []
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            low_text, high_text = token.split("-", maxsplit=1)
            bins.append(IntBin(int(low_text), int(high_text)))
        else:
            value = int(token)
            bins.append(IntBin(value, value))
    if not bins:
        raise ValueError("at least one bin is required")
    for previous, current in itertools.pairwise(bins):
        if previous.high >= current.low:
            raise ValueError(f"bins overlap or are unsorted: {previous.label}, {current.label}")
    return tuple(bins)


def parse_split(spec: str) -> SplitSpec:
    """Parse ``NAME:START:END:PER_STRATUM``."""

    parts = spec.split(":")
    if len(parts) != 4:
        raise ValueError("split must have the form NAME:START:END:PER_STRATUM")
    return SplitSpec(parts[0], int(parts[1]), int(parts[2]), int(parts[3]))


def stable_hash(seed: str, value: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def stratum_for(row: PuzzleRow, config: SelectionConfig) -> tuple[IntBin, IntBin] | None:
    player_moves = len(row.moves) // 2 if len(row.moves) % 2 == 0 else -1
    rating_bin = next((item for item in config.rating_bins if item.contains(row.rating)), None)
    length_bin = next((item for item in config.length_bins if item.contains(player_moves)), None)
    if rating_bin is None or length_bin is None:
        return None
    return rating_bin, length_bin


def select_tasks(rows: Iterable[PuzzleRow], config: SelectionConfig) -> SelectionResult:
    """Select stable, position-disjoint, rating/length-stratified tasks.

    Candidate reservoirs are ranked by puzzle ID. The actual split is assigned by the
    normalized post-setup position, so duplicate positions cannot leak across splits.
    """

    strata = [
        (rating_bin, length_bin)
        for rating_bin in config.rating_bins
        for length_bin in config.length_bins
    ]
    pool_size = max(
        math.ceil(split.per_stratum / split.fraction * config.oversample_factor)
        for split in config.splits
    )
    pools: dict[tuple[str, str], list[tuple[int, int, PuzzleRow]]] = {
        (rating_bin.label, length_bin.label): [] for rating_bin, length_bin in strata
    }
    counts: Counter[str] = Counter()
    sequence = 0

    for row in rows:
        counts["source_rows"] += 1
        if row.popularity < config.min_popularity:
            counts["filtered_popularity"] += 1
            continue
        if row.nb_plays < config.min_plays:
            counts["filtered_plays"] += 1
            continue
        if row.rating_deviation > config.max_rating_deviation:
            counts["filtered_rating_deviation"] += 1
            continue
        if set(row.themes).intersection(config.excluded_themes):
            counts["filtered_excluded_theme"] += 1
            continue
        matched = stratum_for(row, config)
        if matched is None:
            counts["outside_requested_strata"] += 1
            continue

        counts["eligible_rows"] += 1
        key = (matched[0].label, matched[1].label)
        rank = stable_hash(f"{config.seed}:sample", row.puzzle_id)
        heap = pools[key]
        entry = (-rank, -sequence, row)
        sequence += 1
        if len(heap) < pool_size:
            heapq.heappush(heap, entry)
        elif rank < -heap[0][0]:
            heapq.heapreplace(heap, entry)

    candidates: list[tuple[int, PuzzleRow, str, str]] = []
    for (rating_label, length_label), heap in pools.items():
        for negative_rank, _, row in heap:
            candidates.append((-negative_rank, row, rating_label, length_label))
    candidates.sort(key=lambda item: (item[0], item[1].puzzle_id))
    counts["candidate_rows"] = len(candidates)

    selected: dict[str, list[SelectedTask]] = {split.name: [] for split in config.splits}
    selected_counts: Counter[tuple[str, str, str]] = Counter()
    selected_positions: set[str] = set()

    for rank, row, rating_label, length_label in candidates:
        try:
            puzzle = validate_puzzle(row.source_fen, row.moves)
        except ValueError:
            counts["filtered_invalid_line"] += 1
            continue
        if puzzle.normalized_puzzle_fen in selected_positions:
            counts["filtered_duplicate_position"] += 1
            continue

        split_bucket = stable_hash(f"{config.seed}:split", puzzle.normalized_puzzle_fen) % 10_000
        split = next(
            (item for item in config.splits if item.bucket_start <= split_bucket < item.bucket_end),
            None,
        )
        if split is None:
            counts["outside_requested_splits"] += 1
            continue
        cell = (split.name, rating_label, length_label)
        if selected_counts[cell] >= split.per_stratum:
            continue

        metadata = puzzle_metadata(
            row,
            puzzle_fen=puzzle.puzzle_fen,
            normalized_puzzle_fen=puzzle.normalized_puzzle_fen,
            player_moves=puzzle.player_moves,
            split=split.name,
            split_bucket=split_bucket,
            rating_bin=rating_label,
            length_bin=length_label,
        )
        task = task_from_metadata(
            metadata,
            reasoning_budget=config.reasoning_budget,
            tries=config.tries,
        )
        selected[split.name].append(
            SelectedTask(task, row.puzzle_id, rank, rating_label, length_label)
        )
        selected_counts[cell] += 1
        selected_positions.add(puzzle.normalized_puzzle_fen)

    missing: list[str] = []
    for split in config.splits:
        for rating_bin, length_bin in strata:
            cell = (split.name, rating_bin.label, length_bin.label)
            actual = selected_counts[cell]
            if actual != split.per_stratum:
                missing.append(
                    f"{split.name}/{rating_bin.label}/{length_bin.label}: "
                    f"wanted {split.per_stratum}, found {actual}"
                )
    if missing:
        detail = "\n".join(missing)
        raise RuntimeError(
            "candidate reservoir did not fill every requested stratum; increase "
            f"--oversample-factor or relax filters:\n{detail}"
        )

    ordered_tasks = {
        split.name: [item.task for item in stratified_round_robin(selected[split.name])]
        for split in config.splits
    }
    strata_counts: dict[str, dict[str, int]] = {}
    for split in config.splits:
        strata_counts[split.name] = {
            f"rating={rating.label},moves={length.label}": selected_counts[
                (split.name, rating.label, length.label)
            ]
            for rating, length in strata
        }
        counts[f"selected_{split.name}"] = len(ordered_tasks[split.name])
    return SelectionResult(ordered_tasks, dict(sorted(counts.items())), strata_counts)


def stratified_round_robin(tasks: list[SelectedTask]) -> list[SelectedTask]:
    """Order tasks so every short prefix remains approximately stratified."""

    groups: dict[tuple[str, str], list[SelectedTask]] = defaultdict(list)
    for task in tasks:
        groups[(task.rating_bin, task.length_bin)].append(task)
    for group in groups.values():
        group.sort(key=lambda item: (item.sample_rank, item.puzzle_id))

    ordered: list[SelectedTask] = []
    keys = sorted(groups)
    max_size = max((len(group) for group in groups.values()), default=0)
    for index in range(max_size):
        for key in keys:
            if index < len(groups[key]):
                ordered.append(groups[key][index])
    return ordered


def puzzle_metadata(
    row: PuzzleRow,
    *,
    puzzle_fen: str,
    normalized_puzzle_fen: str,
    player_moves: tuple[str, ...],
    split: str,
    split_bucket: int,
    rating_bin: str,
    length_bin: str,
) -> dict[str, Any]:
    """Preserve every source field alongside useful derived fields."""

    return {
        "puzzle_id": row.puzzle_id,
        "source_fen": row.source_fen,
        "moves": list(row.moves),
        "rating": row.rating,
        "rating_deviation": row.rating_deviation,
        "popularity": row.popularity,
        "nb_plays": row.nb_plays,
        "themes": list(row.themes),
        "game_url": row.game_url,
        "opening_tags": list(row.opening_tags),
        "daily_date": row.daily_date,
        "puzzle_fen": puzzle_fen,
        "normalized_puzzle_fen": normalized_puzzle_fen,
        "setup_move": row.moves[0],
        "expected_player_moves": list(player_moves),
        "solution_player_moves": len(player_moves),
        "solution_plies_after_setup": len(row.moves) - 1,
        "side_to_move": "white" if puzzle_fen.split()[1] == "w" else "black",
        "rating_bin": rating_bin,
        "length_bin": length_bin,
        "split": split,
        "split_bucket": split_bucket,
        "source_dataset": "Lichess puzzle database",
        "source_url": LICHESS_SOURCE_URL,
        "source_license": LICHESS_LICENSE,
    }


def task_from_metadata(
    metadata: dict[str, Any],
    *,
    reasoning_budget: int | None,
    tries: int = DEFAULT_TRIES,
) -> Task:
    """Create the AC2 task without exposing difficulty or theme labels to the policy."""

    prompt = position_prompt(
        str(metadata["puzzle_fen"]),
        reasoning_budget=reasoning_budget,
        remaining_tries=tries,
    )
    tags = dict(metadata)
    tags["observation_format"] = OBSERVATION_FORMAT
    tags["tries"] = tries
    if reasoning_budget is not None:
        tags["reasoning_budget"] = reasoning_budget
    return Task(
        input=[Message(role="user", content=prompt)],
        env_params={
            "puzzle_id": metadata["puzzle_id"],
            "source_fen": metadata["source_fen"],
            "moves": metadata["moves"],
            "tries": tries,
            "reasoning_budget": reasoning_budget,
            "metadata": metadata,
        },
        grader_params={
            "puzzle_id": metadata["puzzle_id"],
            "expected_player_moves": metadata["expected_player_moves"],
            "rating": metadata["rating"],
            "rating_bin": metadata["rating_bin"],
            "length_bin": metadata["length_bin"],
            "tries": tries,
        },
        tags=tags,
        description=(
            f"Lichess puzzle {metadata['puzzle_id']} | rating {metadata['rating']} | "
            f"{metadata['solution_player_moves']} player moves"
        ),
    )


@contextmanager
def open_csv(path: Path) -> Iterator[TextIO]:
    """Open either the official Zstandard archive or an uncompressed CSV."""

    if path.suffix != ".zst":
        with path.open(encoding="utf-8", newline="") as handle:
            yield handle
        return

    with (
        path.open("rb") as compressed,
        zstandard.ZstdDecompressor().stream_reader(compressed) as reader,
        io.TextIOWrapper(reader, encoding="utf-8", newline="") as text,
    ):
        yield text


def iter_puzzle_rows(path: Path) -> Iterator[PuzzleRow]:
    """Stream typed rows from the official CSV without materializing the corpus."""

    with open_csv(path) as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        missing = set(EXPECTED_COLUMNS).difference(columns)
        if missing:
            raise ValueError(f"Lichess CSV is missing columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            try:
                yield PuzzleRow.from_csv(row)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid Lichess CSV row {line_number}: {exc}") from exc


def write_selection(
    result: SelectionResult,
    *,
    output_dir: Path,
    dataset_prefix: str,
    dataset_version: str,
    config: SelectionConfig,
    source: dict[str, Any],
) -> Path:
    """Write task JSONL files and one manifest consumed by the registration script."""

    output_dir.mkdir(parents=True, exist_ok=True)
    datasets: dict[str, dict[str, Any]] = {}
    for split, tasks in result.tasks.items():
        dataset_name = f"{dataset_prefix}-{split}-{dataset_version}"
        filename = f"{dataset_name}.jsonl"
        path = output_dir / filename
        with path.open("w", encoding="utf-8") as handle:
            for task in tasks:
                handle.write(json.dumps(task.model_dump(mode="json"), sort_keys=True) + "\n")
        datasets[split] = {
            "name": dataset_name,
            "file": filename,
            "task_count": len(tasks),
            "strata": result.strata[split],
        }

    manifest = {
        "schema_version": 1,
        "source": source,
        "selection": {
            "rating_bins": [item.label for item in config.rating_bins],
            "length_bins": [item.label for item in config.length_bins],
            "splits": [asdict(item) for item in config.splits],
            "seed": config.seed,
            "min_popularity": config.min_popularity,
            "min_plays": config.min_plays,
            "max_rating_deviation": config.max_rating_deviation,
            "excluded_themes": list(config.excluded_themes),
            "reasoning_budget": config.reasoning_budget,
            "tries": config.tries,
            "oversample_factor": config.oversample_factor,
        },
        "counts": result.counts,
        "datasets": datasets,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def load_source_manifest(path: Path, *, archive: Path) -> dict[str, Any]:
    """Load downloader provenance, or compute a minimal local-file record."""

    if path.exists():
        source = json.loads(path.read_text())
        if source.get("sha256"):
            return source
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "url": LICHESS_SOURCE_URL,
        "license": LICHESS_LICENSE,
        "local_file": str(archive),
        "size_bytes": archive.stat().st_size,
        "sha256": digest.hexdigest(),
    }
