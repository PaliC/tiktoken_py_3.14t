from __future__ import annotations

import argparse
import time
from pathlib import Path

import tiktoken


def benchmark_file(path: Path, encoding: tiktoken.Encoding) -> tuple[int, int, float]:
    """Return bytes, tokens, seconds for encoding ``path``."""
    # Use stat() to avoid double-counting encoding conversions for byte size.
    byte_length = path.stat().st_size
    text = path.read_text(encoding="utf-8")
    start = time.perf_counter()
    num_tokens = len(encoding.encode(text))
    end = time.perf_counter()
    return byte_length, num_tokens, end - start


def run(sample_dir: Path, encoding_name: str) -> None:
    encoding = tiktoken.get_encoding(encoding_name)

    files = sorted(p for p in sample_dir.iterdir() if p.is_file())
    if not files:
        raise SystemExit(f"No files found in {sample_dir}")

    print(f"Encoding: {encoding_name}")
    print(f"Sample directory: {sample_dir.resolve()}")
    print()
    total_duration = 0
    for file_path in files:
        bytes_len, tokens, elapsed = benchmark_file(file_path, encoding)
        print(f"{file_path.name}")
        print(f"  size: {bytes_len:,} bytes")
        print(f"  tokens: {tokens:,}")
        print(f"  duration: {elapsed:.3f}s")
        print()
        total_duration += elapsed
    print(f"Total duration: {total_duration:.3f}s")
    return total_duration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark encoding of files under text_samples/"
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path("text_samples"),
        help="directory that contains text samples",
    )
    parser.add_argument(
        "--encoding",
        default="gpt2",
        help="encoding name to use, defaults to gpt2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.samples, args.encoding)


if __name__ == "__main__":
    main()
