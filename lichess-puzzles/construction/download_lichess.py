"""Download the public Lichess puzzle snapshot with an auditable content digest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from .dataset import LICHESS_LICENSE, LICHESS_SOURCE_URL

DEFAULT_OUTPUT = Path("data/raw/lichess_db_puzzle.csv.zst")


def download(url: str, output: Path, *, force: bool) -> dict[str, object]:
    if output.exists() and not force:
        raise FileExistsError(f"{output} already exists; pass --force to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f"{output.name}.part")
    request = urllib.request.Request(url, headers={"User-Agent": "ac2-lichess-puzzles/0.1"})
    digest = hashlib.sha256()
    downloaded = 0
    started = time.monotonic()

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            total = int(response.headers.get("Content-Length", "0"))
            with partial.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
                    digest.update(chunk)
                    downloaded += len(chunk)
                    if downloaded % (32 * 1024 * 1024) < len(chunk):
                        pct = 100 * downloaded / total if total else 0
                        print(f"downloaded {downloaded:,}/{total:,} bytes ({pct:.1f}%)", file=sys.stderr)
            headers = dict(response.headers.items())
        os.replace(partial, output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    return {
        "url": url,
        "license": LICHESS_LICENSE,
        "local_file": str(output),
        "size_bytes": downloaded,
        "sha256": digest.hexdigest(),
        "etag": headers.get("ETag"),
        "last_modified": headers.get("Last-Modified"),
        "download_seconds": round(time.monotonic() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=LICHESS_SOURCE_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    metadata = download(args.url, args.output, force=args.force)
    manifest = args.output.parent / "source_manifest.json"
    manifest.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
