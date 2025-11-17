from __future__ import annotations

from array import array
from bisect import bisect_left
from typing import AbstractSet, Any, Iterable, List, Sequence
import platform
import os
import regex

Rank = int
_RANK_MAX: int = (1 << 32) - 1
_SPACE_BYTES = {ord(" "), ord("\n"), ord("\t")}
# Regex timeout in seconds to prevent catastrophic backtracking.
# Similar to fancyregex's backtrack_limit, this prevents regex operations from taking too long.
_REGEX_TIMEOUT: float = 2.0

import subprocess
import os

MAX_WORKERS = 4


from concurrent.futures import ThreadPoolExecutor

def _byte_pair_merge(ranks: dict[bytes, Rank], piece: bytes) -> list[tuple[int, Rank]]:
    if len(piece) < 2:
        raise ValueError("byte pair merge requires at least two bytes")

    parts: list[list[int | Rank]] = []
    min_rank_value = _RANK_MAX
    min_rank_index = len(piece)
    for i in range(len(piece) - 1):
        rank = ranks.get(piece[i : i + 2], _RANK_MAX)
        parts.append([i, rank])
        if rank < min_rank_value:
            min_rank_value = rank
            min_rank_index = i
    parts.append([len(piece) - 1, _RANK_MAX])
    parts.append([len(piece), _RANK_MAX])

    def get_rank(container: list[list[int | Rank]], idx: int) -> Rank:
        if idx + 3 < len(container):
            start = container[idx][0]
            end = container[idx + 3][0]
            key = piece[start:end]
            return ranks.get(key, _RANK_MAX)
        return _RANK_MAX

    while min_rank_value != _RANK_MAX:
        i = min_rank_index
        if i > 0:
            parts[i - 1][1] = get_rank(parts, i - 1)
        parts[i][1] = get_rank(parts, i)
        parts.pop(i + 1)

        min_rank_value = _RANK_MAX
        for idx, (_, rank) in enumerate(parts[:-1]):
            if rank < min_rank_value:
                min_rank_value = rank
                min_rank_index = idx
    return [(start, rank) for start, rank in parts]  # type: ignore[arg-type]


def byte_pair_encode(piece: bytes, ranks: dict[bytes, Rank]) -> list[Rank]:
    if len(piece) == 1:
        return [ranks[piece]]
    merges = _byte_pair_merge(ranks, piece)
    tokens = []
    for left, right in zip(merges, merges[1:]):
        tokens.append(ranks[piece[left[0] : right[0]]])
    return tokens


def byte_pair_split(piece: bytes, ranks: dict[bytes, Rank]) -> list[bytes]:
    if len(piece) < 2:
        return [piece]
    merges = _byte_pair_merge(ranks, piece)
    return [piece[left[0] : right[0]] for left, right in zip(merges, merges[1:])]


def _decode_last_utf8(data: bytes) -> tuple[str | None, int]:
    max_width = min(4, len(data))
    for width in range(1, max_width + 1):
        start = len(data) - width
        chunk = data[start:]
        try:
            decoded = chunk.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if len(decoded) == 1:
            return decoded, width
    return None, 0


class CoreBPE:
    __slots__ = (
        "encoder",
        "special_tokens_encoder",
        "decoder",
        "special_tokens_decoder",
        "_regex",
        "_special_regex",
        "_sorted_token_bytes",
        "_special_tokens",
        "_chunk_size",
        "_thread_pool",
    )

    def __init__(
        self,
        encoder: dict[bytes, Rank] | Iterable[tuple[bytes, Rank]],
        special_tokens_encoder: dict[str, Rank] | Iterable[tuple[str, Rank]],
        pattern: str,
    ) -> None:
        if not isinstance(encoder, dict):
            encoder = dict(encoder)
        if not isinstance(special_tokens_encoder, dict):
            special_tokens_encoder = dict(special_tokens_encoder)
        self._chunk_size = 50
        self._thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self.encoder = dict(encoder)
        self.special_tokens_encoder = dict(special_tokens_encoder)
        self.decoder = {token: token_bytes for token_bytes, token in self.encoder.items()}
        if len(self.decoder) != len(self.encoder):
            raise ValueError(
                "encoder and decoder must contain the same number of entries; "
                "duplicate token ids detected"
            )
        self.special_tokens_decoder = {
            token: text.encode("utf-8") for text, token in self.special_tokens_encoder.items()
        }

        self._regex = regex.compile(pattern)
        if self.special_tokens_encoder:
            escaped = (regex.escape(token) for token in self.special_tokens_encoder.keys())
            self._special_regex = regex.compile("|".join(escaped))
        else:
            self._special_regex = None

        self._sorted_token_bytes = sorted(self.encoder.keys())
        self._special_tokens = frozenset(self.special_tokens_encoder.keys())

    # ====================
    # Encoding
    # ====================

    def encode_ordinary(self, text: str) -> list[Rank]:
        ret: list[Rank] = []
        match_count = 0
        for match in self._regex.finditer(text, timeout=_REGEX_TIMEOUT):
            match_count += 1
            
            piece = match.group(0).encode("utf-8")
            token = self.encoder.get(piece)
            if token is not None:
                ret.append(token)
            else:
                ret.extend(byte_pair_encode(piece, self.encoder))
        print(f"found {match_count}")
        return ret

    def encode(self, text: str, allowed_special: AbstractSet[str]) -> list[Rank]:
        tokens, _ = self._encode_native(text, allowed_special)
        return tokens

    def encode_to_tiktoken_buffer(self, text: str, allowed_special: AbstractSet[str]) -> array:
        tokens, _ = self._encode_native(text, allowed_special)
        return array("I", tokens)

    def encode_with_unstable(
        self, text: str, allowed_special: AbstractSet[str]
    ) -> tuple[list[Rank], list[list[Rank]]]:
        tokens, completions = self._encode_unstable_native(text, allowed_special)
        completions_list = [list(seq) for seq in sorted(completions)]
        return tokens, completions_list

    def encode_single_token(self, piece: bytes) -> Rank:
        token = self.encoder.get(piece)
        if token is not None:
            return token
        try:
            text_piece = piece.decode("utf-8")
        except UnicodeDecodeError as exc:  # pragma: no cover - parity with rust KeyError
            raise KeyError(piece) from exc
        token = self.special_tokens_encoder.get(text_piece)
        if token is None:
            raise KeyError(piece)
        return token

    def encode_single_piece(self, piece: bytes) -> list[Rank]:
        token = self.encoder.get(piece)
        if token is not None:
            return [token]
        return byte_pair_encode(piece, self.encoder)

    def _encode_bytes(self, data: bytes) -> list[Rank]:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            valid_prefix = data[: exc.start]
            text = valid_prefix.decode("utf-8", errors="strict")
            tokens, last_piece_len = self._encode_native(text, frozenset())
            tokens, last_piece_len = self._increase_last_piece_token_len(tokens, last_piece_len)

            if tokens and last_piece_len > 0:
                unstable_bytes = bytearray(self.decode_bytes(tokens[-last_piece_len:]))
                unstable_bytes.extend(data[exc.start :])
                del tokens[-last_piece_len:]
            else:
                unstable_bytes = bytearray(data[exc.start :])

            if unstable_bytes:
                token = self.encoder.get(bytes(unstable_bytes))
                if token is not None:
                    tokens.append(token)
                else:
                    tokens.extend(byte_pair_encode(bytes(unstable_bytes), self.encoder))
            return tokens
        else:
            return self.encode_ordinary(text)

    # ====================
    # Decoding
    # ====================

    def decode_bytes(self, tokens: Sequence[Rank]) -> bytes:
        output = bytearray()
        for token in tokens:
            if token in self.decoder:
                output.extend(self.decoder[token])
            elif token in self.special_tokens_decoder:
                output.extend(self.special_tokens_decoder[token])
            else:
                raise KeyError(token)
        return bytes(output)

    def decode_single_token_bytes(self, token: Rank) -> bytes:
        if token in self.decoder:
            return self.decoder[token]
        if token in self.special_tokens_decoder:
            return self.special_tokens_decoder[token]
        raise KeyError(token)

    # ====================
    # Miscellaneous
    # ====================

    def token_byte_values(self) -> list[bytes]:
        return list(self._sorted_token_bytes)

    def special_tokens(self) -> set[str]:
        return set(self._special_tokens)

    def encode_with_special_tokens(self, text: str) -> list[Rank]:
        tokens, _ = self._encode_native(text, self._special_tokens)
        return tokens

    # ====================
    # Internal helpers
    # ====================

    def _encode_native(
        self, text: str, allowed_special: AbstractSet[str]
    ) -> tuple[list[Rank], int]:
        allowed_set = {token for token in allowed_special if token in self._special_tokens}
        tokens: list[Rank] = []
        last_piece_token_len = 0
        start = 0

        while True:
            next_special = None
            if self._special_regex is not None and allowed_set:
                start_find = start
                while True:
                    match = self._special_regex.search(text, start_find, timeout=_REGEX_TIMEOUT)
                    if match is None:
                        break
                    piece = text[match.start() : match.end()]
                    if piece in allowed_set:
                        next_special = match
                        break
                    start_find = match.start() + 1

            end = next_special.start() if next_special else len(text)
            matches = list(self._regex.finditer(text[start:end], timeout=_REGEX_TIMEOUT))
            # Only parallelize if we have enough work
            if len(matches) > 0:  # Threshold to avoid overhead
                tokens_chunk, last_piece_token_len = self._encode_native_parallel(matches)
                tokens.extend(tokens_chunk)
            else:
                # Sequential for small workloads
                for match in matches:
                    piece = match.group(0).encode("utf-8")
                    token = self.encoder.get(piece)
                    if token is not None:
                        tokens.append(token)
                        last_piece_token_len = 1
                    else:
                        bpe_tokens = byte_pair_encode(piece, self.encoder)
                        tokens.extend(bpe_tokens)
                        last_piece_token_len = len(bpe_tokens)

            if next_special:
                tokens.append(self.special_tokens_encoder[next_special.group(0)])
                start = next_special.end()
                last_piece_token_len = 0
            else:
                break
        return tokens, last_piece_token_len

    def _encode_native_parallel(self, matches: list) -> tuple[list[Rank], int]:
        if len(matches) < self._chunk_size:
            return self._encode_matches_sequential(matches)
        
        def process_chunk(chunk):
            chunk_tokens = []
            last_len = 0
            
            for match in chunk:
                piece = match.group(0).encode("utf-8")
                token = self.encoder.get(piece)
                
                if token is not None:
                    chunk_tokens.append(token)
                    last_len = 1
                else:
                    bpe_tokens = byte_pair_encode(piece, self.encoder)
                    chunk_tokens.extend(bpe_tokens)
                    last_len = len(bpe_tokens)
            
            return chunk_tokens, last_len
        
        chunks = [matches[i:i + self._chunk_size] 
                  for i in range(0, len(matches), self._chunk_size)]
        
        futures = [self._thread_pool.submit(process_chunk, chunk) 
                   for chunk in chunks]
        
        tokens = []
        last_len = 0
        for future in futures:
            chunk_tokens, token_len = future.result()
            tokens.extend(chunk_tokens)
            last_len = token_len
        
        return tokens, last_len

    def _encode_unstable_native(
        self, text: str, allowed_special: AbstractSet[str]
    ) -> tuple[list[Rank], set[tuple[Rank, ...]]]:
        tokens, last_piece_len = self._encode_native(text, allowed_special)
        if last_piece_len == 0:
            return tokens, set()
        tokens, last_piece_len = self._increase_last_piece_token_len(tokens, last_piece_len)

        unstable_segment = self.decode_bytes(tokens[-last_piece_len:])
        tokens = tokens[:-last_piece_len]

        completions: set[tuple[Rank, ...]] = set()
        if not unstable_segment:
            return tokens, completions

        point = bisect_left(self._sorted_token_bytes, unstable_segment)
        while point < len(self._sorted_token_bytes):
            token_bytes = self._sorted_token_bytes[point]
            if not token_bytes.startswith(unstable_segment):
                break
            completions.add((self.encoder[token_bytes],))
            point += 1

        for i in range(1, len(unstable_segment)):
            prefix = unstable_segment[:i]
            suffix = unstable_segment[i:]
            point = bisect_left(self._sorted_token_bytes, suffix)
            while point < len(self._sorted_token_bytes):
                token_bytes = self._sorted_token_bytes[point]
                if not token_bytes.startswith(suffix):
                    break
                possibility = prefix + token_bytes
                try:
                    encoded = self.encode_ordinary(possibility.decode("utf-8"))
                except UnicodeDecodeError:
                    encoded = byte_pair_encode(possibility, self.encoder)
                seq: list[Rank] = []
                seq_len = 0
                for token in encoded:
                    seq.append(token)
                    seq_len += len(self.decoder[token])
                    if seq_len >= len(unstable_segment):
                        break
                completions.add(tuple(seq))
                point += 1

        if len(unstable_segment) > 1:
            last_char, trailing_len = _decode_last_utf8(unstable_segment)
            if (
                trailing_len
                and len(unstable_segment) - trailing_len > 0
                and last_char is not None
                and last_char.isspace()
            ):
                first_part = byte_pair_encode(unstable_segment[:-trailing_len], self.encoder)
                second_part = byte_pair_encode(unstable_segment[-trailing_len:], self.encoder)
                completions.add(tuple(first_part + second_part))

        return tokens, completions

    def _increase_last_piece_token_len(
        self, tokens: list[Rank], last_piece_len: int
    ) -> tuple[list[Rank], int]:
        if last_piece_len == 0 or last_piece_len > len(tokens):
            return tokens, last_piece_len

        def token_is_all_space(token_id: Rank) -> bool:
            token_bytes = self.decoder.get(token_id)
            if token_bytes is None:
                return False
            return all(byte in _SPACE_BYTES for byte in reversed(token_bytes))

        if token_is_all_space(tokens[-last_piece_len]):
            while last_piece_len < len(tokens) and token_is_all_space(tokens[-last_piece_len - 1]):
                last_piece_len += 1
        return tokens, last_piece_len

    def __del__(self):
        self._thread_pool.shutdown(wait=False)


__all__ = ["CoreBPE", "byte_pair_encode", "byte_pair_split"]
