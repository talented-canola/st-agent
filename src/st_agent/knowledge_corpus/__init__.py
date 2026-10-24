"""Versioned external-original corpus; no model-generated knowledge."""

from .contracts import CorpusError, SourceSpec
from .parsers import parse_source
from .store import CorpusStore

__all__ = ["CorpusError", "CorpusStore", "SourceSpec", "parse_source"]
