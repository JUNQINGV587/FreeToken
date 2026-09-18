"""The tokenizer-vs-embedding vocab guard.

A tokenizer that can emit ids past the embedding table makes the text embedding gather
read out of bounds (no mask on that path), which returns whatever memory sits behind the
table: silent corruption, not a crash. Reported numbers come from the mini test models
(``vocab_size`` 32768 vs a tokenizer topping out at 248076).
"""

from __future__ import annotations

import freetoken.utils.hf as hf
import pytest


class _Tok:
    def __init__(self, vocab: dict[str, int], added: dict[str, int] | None = None) -> None:
        self._vocab = vocab
        self.added_tokens_encoder = dict(added or {})

    def get_vocab(self) -> dict[str, int]:
        return self._vocab


class _Cfg:
    def __init__(self, **fields: object) -> None:
        self.__dict__.update(fields)


class _Recorder:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)


@pytest.fixture
def logged(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    recorder = _Recorder()
    monkeypatch.setattr(hf, "logger", recorder)
    return recorder


def test_max_token_id_includes_added_tokens() -> None:
    tok = _Tok({"a": 3, "b": 32767}, {"<pad>": 248076})
    assert hf.tokenizer_max_token_id(tok) == 248076


def test_max_token_id_on_base_vocab_alone() -> None:
    assert hf.tokenizer_max_token_id(_Tok({"a": 0, "b": 11})) == 11


def test_token_ids_union_base_and_added_without_duplicates() -> None:
    tok = _Tok({"a": 0, "b": 11, "<pad>": 20}, {"<pad>": 20, "<im_end>": 248076})
    assert hf.tokenizer_token_ids(tok) == [0, 11, 20, 248076]


def test_config_vocab_size_reads_nested_text_config() -> None:
    assert hf.config_vocab_size(_Cfg(text_config=_Cfg(vocab_size=248320))) == 248320


def test_config_vocab_size_reads_top_level() -> None:
    assert hf.config_vocab_size(_Cfg(vocab_size=32768)) == 32768


def test_config_vocab_size_is_none_when_absent() -> None:
    assert hf.config_vocab_size(_Cfg()) is None


def test_overflow_is_reported(logged: _Recorder) -> None:
    tok = _Tok({str(i): i for i in range(32768)}, {"<im_end>": 248076})
    largest = hf.warn_on_tokenizer_vocab_overflow(tok, _Cfg(text_config=_Cfg(vocab_size=32768)), "mini")
    assert largest == 248076
    assert len(logged.errors) == 1
    assert "248076" in logged.errors[0] and "32768" in logged.errors[0]
    assert "mini" in logged.errors[0]


def test_covering_vocabulary_is_silent(logged: _Recorder) -> None:
    tok = _Tok({str(i): i for i in range(1000)}, {"<im_end>": 248076})
    largest = hf.warn_on_tokenizer_vocab_overflow(tok, _Cfg(text_config=_Cfg(vocab_size=248320)))
    assert largest == 248076
    assert logged.errors == []


def test_unknown_vocab_size_is_silent(logged: _Recorder) -> None:
    tok = _Tok({"a": 5}, {"<pad>": 999})
    hf.warn_on_tokenizer_vocab_overflow(tok, _Cfg())
    assert logged.errors == []
