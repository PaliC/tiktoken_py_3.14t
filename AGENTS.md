# Repository Guidelines

## Project Structure & Module Organization
Python APIs live under `tiktoken/` (e.g., `core.py` and registry helpers) and should remain the only import surface for clients. The Rust tokenizer resides in `src/lib.rs` with bindings in `src/py.rs`; keep interfaces mirrored when changing signatures or error semantics. Encoding definitions and extension hooks sit in the namespace package `tiktoken_ext/` (leave it without `__init__.py`). Tests plus fixtures are in `tests/`, while tooling such as `scripts/benchmark.py` and `scripts/redact.py` stays under `scripts/`.

## Build, Test, and Development Commands
- `python -m pip install -e .[blobfile]` — compiles the Rust extension via `setuptools-rust` and installs optional blobfile helpers for tokenizer assets.
- `pytest tests --import-mode=append` — mirrors the CI invocation from `pyproject.toml`; run before every PR and when regenerating encodings.
- `cargo test --lib` — exercises the Edition 2024 Rust crate; required after touching `_byte_pair_merge` or decoder logic.
- `python scripts/benchmark.py --encoder cl100k_base --text sample.txt` — quick regression check for throughput-sensitive changes.

## Coding Style & Naming Conventions
- Python: 4-space indent, `from __future__ import annotations`, exhaustive type hints, `snake_case` functions, `PascalCase` classes, docstrings with short examples.
- Keep public names descriptive (`cl100k_base`, `o200k_base`) and guard new special tokens with meaningful identifiers.
- Rust: run `cargo fmt` and `cargo clippy --all-targets --all-features`; document any unsafe blocks and follow the FxHashMap-based patterns already in `src/lib.rs`.

## Testing Guidelines
Add deterministic pytest cases next to similar topics (`tests/test_encoding.py` for tokenizer behavior, `tests/test_simple_public.py` for API surface). Property-based checks already use Hypothesis with `MAX_EXAMPLES` from `tests/test_helpers.py`; extend those strategies instead of duplicating random loops. Mention focused commands such as `pytest tests -k surrogate` in PRs, and include failing inputs when filing issues.

## Commit & Pull Request Guidelines
History prefers short, imperative subjects with optional scopes and PR references (e.g., `chore: update dependencies (#449)` or `Add GPT-5 model support with o200k_base encoding (#440)`). Keep subjects ≤72 characters, expand motivation plus risk in the body, and reference GitHub issues by number. Every PR should list what changed, how it was tested (`pytest…`, `cargo test`, benchmarks), and whether registry consumers must take action. Attach before/after measurements or screenshots when touching performance docs or `perf.svg`.
