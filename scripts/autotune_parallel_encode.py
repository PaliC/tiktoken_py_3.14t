#!/usr/bin/env python
from __future__ import annotations

"""Autotune CoreBPE parallel chunking parameters via encode benchmarks."""

import argparse
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for restricted environments
    class _SimpleTqdm:
        def __init__(self, iterable, total: int | None = None, desc: str | None = None, unit: str | None = None):
            self._iterable = iterable
            self._total = total
            self._count = 0
            self._desc = desc or ""
            self._unit = unit or ""
            self._postfix = ""

        def __iter__(self):
            for item in self._iterable:
                self._count += 1
                self._print_status()
                yield item
            print()

        def set_postfix(self, values: dict[str, object] | None = None):
            if values:
                formatted = ", ".join(f"{key}={value}" for key, value in values.items())
                self._postfix = f" {formatted}"
            else:
                self._postfix = ""
            self._print_status()

        def _print_status(self):
            if not self._total:
                status = f"{self._count}"
            else:
                status = f"{self._count}/{self._total}"
            prefix = f"{self._desc} " if self._desc else ""
            suffix = f" {self._unit}" if self._unit else ""
            print(f"\r{prefix}{status}{suffix}{self._postfix}", end="", flush=True)

    def tqdm(iterable, total: int | None = None, desc: str | None = None, unit: str | None = None):
        return _SimpleTqdm(iterable, total=total, desc=desc, unit=unit)

import tiktoken
from tiktoken import _tiktoken
from tiktoken import registry as tk_registry


@dataclass
class BenchmarkResult:
    chunk_size: int
    max_workers: int
    durations: list[float]
    tokens_processed: int
    text_bytes: int

    @property
    def mean_seconds(self) -> float:
        return statistics.fmean(self.durations)

    @property
    def std_seconds(self) -> float:
        return statistics.stdev(self.durations) if len(self.durations) > 1 else 0.0

    @property
    def tokens_per_second(self) -> float:
        return self.tokens_processed / self.mean_seconds

    @property
    def mb_per_second(self) -> float:
        # Convert bytes/sec to mebibytes/sec for readability
        return (self.text_bytes / self.mean_seconds) / (1024 * 1024)


def _parse_positive_list(values: Sequence[int], label: str) -> list[int]:
    parsed = [int(v) for v in values]
    if not parsed:
        raise ValueError(f"At least one value is required for {label}.")
    if any(v <= 0 for v in parsed):
        raise ValueError(f"{label} entries must be positive integers: {parsed}")
    return parsed


def _resolve_text_files(paths: Sequence[Path]) -> list[Path]:
    if not paths:
        raise ValueError("Provide at least one --text path.")
    resolved: list[Path] = []
    for path in paths:
        if path.is_dir():
            resolved.extend(sorted(p for p in path.rglob("*.txt") if p.is_file()))
        elif path.is_file():
            resolved.append(path)
        else:
            raise FileNotFoundError(f"{path} does not exist.")
    if not resolved:
        raise ValueError("No text files found. Ensure directories contain .txt files.")
    return resolved


def load_text(sample_paths: Sequence[Path], max_chars: int | None, repeat: int) -> str:
    text_parts: list[str] = []
    for file_path in _resolve_text_files(sample_paths):
        text_parts.append(file_path.read_text(encoding="utf-8"))
    text = "".join(text_parts)
    if max_chars is not None:
        text = text[:max_chars]
    if repeat > 1:
        text *= repeat
    return text


def configure_encoding(encoding_name: str, chunk_size: int, max_workers: int) -> tiktoken.Encoding:
    # Drop any cached Encoding so a fresh CoreBPE instance is created.
    tk_registry.ENCODINGS.pop(encoding_name, None)
    _tiktoken.MAX_WORKERS = max_workers

    encoding = tiktoken.get_encoding(encoding_name)
    encoding._core_bpe._chunk_size = chunk_size
    return encoding


def benchmark_configuration(
    *,
    encoding_name: str,
    text: str,
    text_bytes: int,
    chunk_size: int,
    max_workers: int,
    warmup_runs: int,
    trials: int,
) -> BenchmarkResult:
    encoding = configure_encoding(encoding_name, chunk_size, max_workers)

    try:
        for _ in range(warmup_runs):
            encoding.encode(text)

        durations: list[float] = []
        tokens_seen = 0
        for _ in range(trials):
            start = time.perf_counter()
            tokens = encoding.encode(text)
            elapsed = time.perf_counter() - start
            durations.append(elapsed)
            tokens_seen = len(tokens)

        return BenchmarkResult(
            chunk_size=chunk_size,
            max_workers=max_workers,
            durations=durations,
            tokens_processed=tokens_seen,
            text_bytes=text_bytes,
        )
    finally:
        # Explicitly stop the worker threads so we don't leak pools between runs.
        encoding._core_bpe._thread_pool.shutdown(wait=True)


def run_autotune(
    *,
    chunk_sizes: Iterable[int],
    max_workers: Iterable[int],
    encoding_name: str,
    sample_paths: Sequence[Path],
    max_chars: int | None,
    repeat: int,
    warmup_runs: int,
    trials: int,
) -> list[BenchmarkResult]:
    chunk_sizes = list(chunk_sizes)
    max_workers = list(max_workers)

    text = load_text(sample_paths, max_chars, repeat)
    if not text:
        raise ValueError("Sample text is empty. Provide a larger file or reduce truncation.")
    text_bytes = len(text.encode("utf-8"))

    results: list[BenchmarkResult] = []
    total_runs = len(chunk_sizes) * len(max_workers)
    progress = tqdm(
        ((chunk_size, workers) for chunk_size in chunk_sizes for workers in max_workers),
        total=total_runs,
        desc="Benchmarking configs",
        unit="config",
    )
    for chunk_size, workers in progress:
        progress.set_postfix({"chunk_size": chunk_size, "max_workers": workers})
        results.append(
            benchmark_configuration(
                encoding_name=encoding_name,
                text=text,
                text_bytes=text_bytes,
                chunk_size=chunk_size,
                max_workers=workers,
                warmup_runs=warmup_runs,
                trials=trials,
            )
        )
    return results


def format_summary(results: Sequence[BenchmarkResult]) -> str:
    best = max(results, key=lambda r: r.tokens_per_second)
    header = f"{'chunk':>8} {'workers':>8} {'mean(s)':>10} {'std(s)':>9} {'tok/s':>12} {'MiB/s':>10} {'delta%':>8}"
    lines = [header, "-" * len(header)]
    for result in sorted(results, key=lambda r: (r.chunk_size, r.max_workers)):
        delta = (result.tokens_per_second / best.tokens_per_second - 1.0) * 100
        lines.append(
            f"{result.chunk_size:8d} {result.max_workers:8d} {result.mean_seconds:10.4f} "
            f"{result.std_seconds:9.4f} {result.tokens_per_second:12.0f} "
            f"{result.mb_per_second:10.2f} {delta:8.2f}"
        )
    lines.append("")
    lines.append(
        "Best configuration: chunk_size={} max_workers={} {:.0f} tokens/s ({:.2f} MiB/s)".format(
            best.chunk_size,
            best.max_workers,
            best.tokens_per_second,
            best.mb_per_second,
        )
    )
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Autotune CoreBPE parallel parameters by benchmarking encode_ordinary()."
    )
    parser.add_argument(
        "--encoder",
        default="cl100k_base",
        help="Encoding name passed to tiktoken.get_encoding (default: cl100k_base)",
    )
    parser.add_argument(
        "--text",
        type=Path,
        nargs="+",
        default=[Path("text_samples/warandpeace.txt")],
        help=(
            "One or more text files or directories (directories will use all *.txt files) "
            "used for the benchmark."
        ),
    )
    parser.add_argument(
        "--chunk-sizes",
        type=int,
        nargs="+",
        default=[16, 32, 50, 64, 96, 128, 192, 256],
        help="List of chunk sizes to evaluate",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        nargs="+",
        default=[1, 2, 4, 6, 8],
        help="List of thread counts to evaluate",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Number of warmup runs per configuration (default: 1)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of timed runs per configuration (default: 3)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=200_000,
        help="Trim the input text to this many characters before repeating (default: 200k)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat the loaded text N times to scale the workload (default: 1)",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    try:
        chunk_sizes = _parse_positive_list(args.chunk_sizes, "chunk sizes")
        worker_counts = _parse_positive_list(args.max_workers, "max workers")
    except ValueError as exc:
        parser.error(str(exc))

    if args.warmup_runs < 0 or args.trials <= 0:
        parser.error("Warmup runs must be >= 0 and trials must be > 0.")
    if args.max_chars is not None and args.max_chars <= 0:
        parser.error("--max-chars must be positive or omitted.")
    if args.repeat <= 0:
        parser.error("--repeat must be positive.")

    results = run_autotune(
        chunk_sizes=chunk_sizes,
        max_workers=worker_counts,
        encoding_name=args.encoder,
        sample_paths=args.text,
        max_chars=args.max_chars,
        repeat=args.repeat,
        warmup_runs=args.warmup_runs,
        trials=args.trials,
    )

    print()
    print(format_summary(results))


if __name__ == "__main__":
    main()
