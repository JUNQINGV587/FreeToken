"""A stop string must match even when it spans more tokens than it has characters.

The scheduler used to decode only the last (longest stop's char count + 1) tokens, so a
multi-byte stop -- an emoji is 4 UTF-8 bytes, i.e. up to 4 byte-level BPE tokens -- could
be generated straight past its own marker: the window opened inside the character and the
decoded window never contained the stop. This pins the incremental decode that replaced
that bounded suffix.

``FragmentTokenizer`` emulates a byte-level BPE ``decode``: a slice that starts inside a
multi-byte character decodes to U+FFFD (``errors="replace"``), which is the behaviour the
detokenizer's own ``endswith("\\ufffd")`` guard is written against.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.scheduler.scheduler import Scheduler

STOP = "\U0001F9EC"  # U+1F9EC, 4 UTF-8 bytes, split into 3 tokens below
PROMPT_LEN = 2  # input_ids[:PROMPT_LEN] is the prompt


class FragmentTokenizer:
    def __init__(self, fragments: dict[int, bytes]) -> None:
        self.fragments = fragments

    def decode(self, ids) -> str:
        raw = b"".join(self.fragments[int(i)] for i in ids)
        return raw.decode("utf-8", errors="replace")


def _req(input_ids: list[int], stops: list[str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        max_device_len=PROMPT_LEN,
        output_len=0,
        can_decode=True,
        sampling_params=SimpleNamespace(stop_strs=stops if stops is not None else [STOP]),
        stop_decode_status=None,
    )


def _match(req: SimpleNamespace) -> str | None:
    scheduler = SimpleNamespace(
        tokenizer=FragmentTokenizer({1: b"\xf0\x9f", 2: b"\xa7", 3: b"\xac"}),
        eos_token_ids=frozenset(),
    )
    return Scheduler._match_stop_str(scheduler, req)


def test_stop_spanning_three_tokens_is_matched():
    # 3 tokens carry one character: wider than max_chars + 1 = 2, which the old
    # last-(max_chars+1)-tokens window could not cover.
    assert _match(_req([0, 0, 1, 2, 3])) == STOP


def test_partial_character_is_not_matched_yet():
    assert _match(_req([0, 0, 1, 2])) is None


def test_absent_stop_is_not_matched():
    assert _match(_req([0, 0, 1, 2, 3], stops=["\u26a0"])) is None
