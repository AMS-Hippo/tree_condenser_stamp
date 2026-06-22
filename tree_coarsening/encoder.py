"""Immutable rule metadata and encoder artifact interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable as HashableABC, Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import Any, Hashable

import networkx as nx

from .exceptions import ConfigurationError, FittingSizeError, LabelMetadataError

MatchingLabel = Hashable


class _WriteOnceArtifact:
    """Prevent fitted semantic state from being reassigned after construction."""

    def __setattr__(self, name: str, value: Any) -> None:
        if name in vars(self):
            raise AttributeError(
                f"{type(self).__name__} artifacts are semantically immutable; "
                f"cannot reassign {name!r}."
            )
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if name in vars(self):
            raise AttributeError(
                f"{type(self).__name__} artifacts are semantically immutable; "
                f"cannot delete {name!r}."
            )
        object.__delattr__(self, name)


def _freeze(value: Any) -> Any:
    """Recursively isolate fitted rule metadata from caller mutation."""

    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class EncodingRule:
    """One ordered, method-independent schema-1 contraction rule."""

    rule_index: int
    operation: str
    output_label: MatchingLabel
    output_fitting_size: int
    pattern: Mapping[str, Any] = field(default_factory=dict)
    parameter_names: tuple[str, ...] = ()
    score: float | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.rule_index, int)
            or isinstance(self.rule_index, bool)
            or self.rule_index < 0
        ):
            raise ConfigurationError(
                f"rule_index must be a nonnegative integer; got {self.rule_index!r}."
            )
        if not isinstance(self.operation, str) or not self.operation:
            raise ConfigurationError("operation must be a nonempty string.")
        if not isinstance(self.output_label, HashableABC):
            raise LabelMetadataError(f"output_label must be hashable; got {self.output_label!r}.")
        try:
            hash(self.output_label)
        except Exception as exc:
            raise LabelMetadataError(
                f"output_label cannot be hashed reliably: {self.output_label!r}."
            ) from exc
        if (
            not isinstance(self.output_fitting_size, int)
            or isinstance(self.output_fitting_size, bool)
            or self.output_fitting_size <= 0
        ):
            raise FittingSizeError(
                "output_fitting_size must be a positive non-Boolean integer; "
                f"got {self.output_fitting_size!r}."
            )
        if not isinstance(self.pattern, Mapping):
            raise ConfigurationError("pattern must be a mapping.")
        if any(not isinstance(key, str) for key in self.pattern):
            raise ConfigurationError("pattern keys must be strings.")
        if not isinstance(self.parameter_names, tuple):
            raise ConfigurationError("parameter_names must be a tuple of strings.")
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ConfigurationError("parameter_names must contain nonempty strings.")
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ConfigurationError("parameter_names must be unique.")
        if self.score is not None:
            if (
                not isinstance(self.score, Real)
                or isinstance(self.score, bool)
                or not isfinite(float(self.score))
            ):
                raise ConfigurationError(f"score must be finite or None; got {self.score!r}.")
            object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "pattern", _freeze(dict(self.pattern)))


class Vocabulary:
    """Finite read-only mapping from stage output labels to fitting sizes."""

    __slots__ = ("_labels", "_sizes")

    def __setattr__(self, name: str, value: Any) -> None:
        if hasattr(self, name):
            raise AttributeError(f"Vocabulary is read-only; cannot reassign {name!r}.")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if hasattr(self, name):
            raise AttributeError(f"Vocabulary is read-only; cannot delete {name!r}.")
        object.__delattr__(self, name)

    def __init__(self, rules: Sequence[EncodingRule] = ()) -> None:
        labels: list[MatchingLabel] = []
        sizes: dict[MatchingLabel, int] = {}
        for expected_index, rule in enumerate(rules):
            if not isinstance(rule, EncodingRule):
                raise ConfigurationError("rules must contain EncodingRule values.")
            if rule.rule_index != expected_index:
                raise ConfigurationError(
                    "rule indices must be consecutive in tuple order: "
                    f"expected {expected_index}, got {rule.rule_index}."
                )
            previous = sizes.get(rule.output_label)
            if previous is None:
                labels.append(rule.output_label)
                sizes[rule.output_label] = rule.output_fitting_size
            elif previous != rule.output_fitting_size:
                raise FittingSizeError(
                    f"output label {rule.output_label!r} has conflicting fitting sizes "
                    f"{previous} and {rule.output_fitting_size}."
                )
        self._labels = tuple(labels)
        self._sizes = MappingProxyType(sizes)

    @property
    def labels(self) -> tuple[MatchingLabel, ...]:
        return self._labels

    def has_label(self, label: MatchingLabel) -> bool:
        return label in self._sizes

    def fitting_size(self, label: MatchingLabel) -> int:
        try:
            return self._sizes[label]
        except KeyError as exc:
            raise LabelMetadataError(f"stage vocabulary has no label {label!r}.") from exc

    def as_dict(self) -> dict[MatchingLabel, int]:
        return dict(self._sizes)

    def __len__(self) -> int:
        return len(self._labels)

    def __contains__(self, label: MatchingLabel) -> bool:
        return label in self._sizes

    def __iter__(self):
        return iter(self._labels)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Vocabulary)
            and self._labels == other._labels
            and self._sizes == other._sizes
        )

    def __repr__(self) -> str:
        return f"Vocabulary({dict(self._sizes)!r})"


class TreeEncoder(_WriteOnceArtifact, ABC):
    """Fitted schema-1 encoder artifact."""

    def __init__(self, *, model_id: str, rules: Sequence[EncodingRule]) -> None:
        if not isinstance(model_id, str) or not model_id:
            raise ConfigurationError("model_id must be a nonempty string.")
        self._model_id = model_id
        self._rules = tuple(rules)
        self._vocab = Vocabulary(self._rules)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def rules(self) -> tuple[EncodingRule, ...]:
        return self._rules

    @property
    def vocab(self) -> Vocabulary:
        return self._vocab

    @abstractmethod
    def transform(
        self,
        graph: nx.DiGraph,
        *,
        validate: str | bool = "full",
    ) -> nx.DiGraph:
        """Transform one raw or schema-1 encoded graph without mutating it."""

    def encode(
        self,
        graph: nx.DiGraph,
        *,
        validate: str | bool = "full",
    ) -> nx.DiGraph:
        """Compatibility alias for :meth:`transform`."""

        return self.transform(graph, validate=validate)
