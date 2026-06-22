"""Occurrence-specific exact structural types for schema 1.0.

Matching labels are deliberately opaque. All geometry required for rewiring
and decoding lives in :class:`CompositeType` occurrences.
"""

from __future__ import annotations

from collections.abc import Hashable as HashableABC, Iterator
from dataclasses import dataclass
from typing import Any, Hashable, Literal, TypeAlias

from .exceptions import ExactTypeError

MatchingLabel: TypeAlias = Hashable
BaseType: TypeAlias = tuple[Literal["base"], str]
AttachMap: TypeAlias = tuple[int, ...]


def _require_hashable(value: Any, *, what: str) -> None:
    if not isinstance(value, HashableABC):
        raise ExactTypeError(f"{what} must be hashable; got {value!r}.")
    try:
        hash(value)
    except Exception as exc:
        raise ExactTypeError(f"{what} cannot be hashed reliably: {value!r}.") from exc


def base_type(raw_label: str) -> BaseType:
    """Return the exact type of one raw node label."""

    if not isinstance(raw_label, str):
        raise ExactTypeError(f"raw base labels must be strings; got {raw_label!r}.")
    return ("base", raw_label)


def is_base_type(value: Any) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and value[0] == "base"
        and isinstance(value[1], str)
    )


@dataclass(frozen=True, slots=True)
class CompositeType:
    """Exact immutable recipe for one contracted occurrence."""

    model_id: str
    label: MatchingLabel
    parent: tuple[int, ...]
    components: tuple["ExactType", ...]
    attach: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ExactTypeError("CompositeType.model_id must be a nonempty string.")
        _require_hashable(self.label, what="CompositeType.label")
        if not isinstance(self.parent, tuple):
            raise ExactTypeError("CompositeType.parent must be a tuple.")
        if not isinstance(self.components, tuple):
            raise ExactTypeError("CompositeType.components must be a tuple.")
        if not isinstance(self.attach, tuple):
            raise ExactTypeError("CompositeType.attach must be a tuple.")

        n = len(self.parent)
        if n == 0:
            raise ExactTypeError("CompositeType requires at least one component.")
        if len(self.components) != n:
            raise ExactTypeError(
                "CompositeType.parent and CompositeType.components must have equal length."
            )
        for i, parent_i in enumerate(self.parent):
            if (
                not isinstance(parent_i, int)
                or isinstance(parent_i, bool)
                or parent_i < -1
                or parent_i >= n
                or parent_i == i
            ):
                raise ExactTypeError(f"invalid CompositeType.parent[{i}]={parent_i!r}.")
        self._check_parent_acyclic()

        for i, component in enumerate(self.components):
            if not is_exact_type(component):
                raise ExactTypeError(f"component {i} is not a valid exact type: {component!r}.")
        for i, value in enumerate(self.attach):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ExactTypeError(
                    f"CompositeType.attach[{i}] must be an integer; got {value!r}."
                )

        expected = sum(
            exact_root_count(self.components[i])
            for i, parent_i in enumerate(self.parent)
            if parent_i != -1
        )
        if len(self.attach) != expected:
            raise ExactTypeError(
                f"CompositeType.attach has length {len(self.attach)}; expected {expected}."
            )

        cursor = 0
        for i, parent_i in enumerate(self.parent):
            if parent_i == -1:
                continue
            width = exact_root_count(self.components[i])
            parent_size = exact_site_count(self.components[parent_i])
            piece = self.attach[cursor : cursor + width]
            cursor += width
            bad = tuple(q for q in piece if q < 0 or q >= parent_size)
            if bad:
                raise ExactTypeError(
                    f"component {i} attaches to invalid sites {bad!r}; "
                    f"parent component {parent_i} has {parent_size} sites."
                )

    @property
    def n_components(self) -> int:
        return len(self.components)

    @property
    def root_positions(self) -> tuple[int, ...]:
        return tuple(i for i, parent_i in enumerate(self.parent) if parent_i == -1)

    @property
    def site_count(self) -> int:
        return sum(exact_site_count(component) for component in self.components)

    @property
    def root_count(self) -> int:
        return sum(
            exact_root_count(self.components[i])
            for i, parent_i in enumerate(self.parent)
            if parent_i == -1
        )

    def attachment_slice(self, component_index: int) -> AttachMap:
        if component_index < 0 or component_index >= self.n_components:
            raise IndexError(component_index)
        if self.parent[component_index] == -1:
            return ()
        cursor = 0
        for i, parent_i in enumerate(self.parent):
            if parent_i == -1:
                continue
            width = exact_root_count(self.components[i])
            if i == component_index:
                return tuple(self.attach[cursor : cursor + width])
            cursor += width
        raise AssertionError("unreachable")

    def attachment_slices(self) -> tuple[AttachMap, ...]:
        out: list[AttachMap] = []
        cursor = 0
        for i, parent_i in enumerate(self.parent):
            if parent_i == -1:
                out.append(())
                continue
            width = exact_root_count(self.components[i])
            out.append(tuple(self.attach[cursor : cursor + width]))
            cursor += width
        return tuple(out)

    def _check_parent_acyclic(self) -> None:
        state = bytearray(len(self.parent))
        for start in range(len(self.parent)):
            if state[start] == 2:
                continue
            path: list[int] = []
            current = start
            while current != -1 and state[current] == 0:
                state[current] = 1
                path.append(current)
                current = self.parent[current]
            if current != -1 and state[current] == 1:
                raise ExactTypeError("CompositeType.parent contains a cycle.")
            for item in path:
                state[item] = 2


ExactType: TypeAlias = BaseType | CompositeType


def is_exact_type(value: Any) -> bool:
    return is_base_type(value) or isinstance(value, CompositeType)


def exact_type_label(value: ExactType) -> MatchingLabel:
    if is_base_type(value):
        return value[1]
    if isinstance(value, CompositeType):
        return value.label
    raise ExactTypeError(f"not an exact type: {value!r}.")


def exact_site_count(value: ExactType) -> int:
    if is_base_type(value):
        return 1
    if isinstance(value, CompositeType):
        return value.site_count
    raise ExactTypeError(f"not an exact type: {value!r}.")


def exact_root_count(value: ExactType) -> int:
    if is_base_type(value):
        return 1
    if isinstance(value, CompositeType):
        return value.root_count
    raise ExactTypeError(f"not an exact type: {value!r}.")


def iter_exact_types(value: ExactType) -> Iterator[ExactType]:
    stack: list[ExactType] = [value]
    while stack:
        current = stack.pop()
        yield current
        if isinstance(current, CompositeType):
            stack.extend(reversed(current.components))


def iter_base_labels(value: ExactType) -> Iterator[str]:
    """Yield raw base labels in exact site order without recursive calls."""

    stack: list[ExactType] = [value]
    while stack:
        current = stack.pop()
        if is_base_type(current):
            yield current[1]
        elif isinstance(current, CompositeType):
            stack.extend(reversed(current.components))
        else:  # pragma: no cover - callers validate exact types first
            raise ExactTypeError(f"not an exact type: {current!r}.")


def exact_type_labels(value: ExactType) -> frozenset[MatchingLabel]:
    return frozenset(exact_type_label(item) for item in iter_exact_types(value))


def exact_type_model_ids(value: ExactType) -> frozenset[str]:
    return frozenset(
        item.model_id for item in iter_exact_types(value) if isinstance(item, CompositeType)
    )
