"""Public exception hierarchy for tree coarsening."""


class TreeCoarseningError(Exception):
    """Base class for package errors."""


class ConfigurationError(TreeCoarseningError):
    """Invalid constructor, rule, or artifact configuration."""


class InternalInvariantError(TreeCoarseningError):
    """A package-internal invariant was violated."""


class NotFittedError(TreeCoarseningError):
    """A fitted operation was requested before fitting."""


class ValidationError(TreeCoarseningError):
    """Input data violates the schema or an operation precondition."""


class GraphSchemaError(ValidationError):
    """Graph-level schema metadata is missing or malformed."""


class TreeStructureError(ValidationError):
    """A graph is not a directed rooted tree."""


class LabelMetadataError(ValidationError):
    """Matching-label metadata is malformed."""


class FittingSizeError(ValidationError):
    """Label-level fitting-size metadata is malformed or inconsistent."""


class ExactTypeError(ValidationError):
    """Occurrence-specific exact structure is malformed."""


class AttachmentError(ValidationError):
    """An attachment map is malformed or geometrically invalid."""


class ProvenanceError(ValidationError):
    """Raw UID or attribute provenance is malformed or inconsistent."""


class StageOrderError(ValidationError):
    """Stage lineage or decode order is invalid."""


class TypeOwnershipError(ValidationError):
    """A decoder does not own an exact structural type."""


class DecodeSelectionError(TreeCoarseningError):
    """A partial-decoding selector is invalid."""


class TargetNotFoundError(DecodeSelectionError):
    """A requested partial-decoding target was not found."""


class BoundaryExpansionError(TreeCoarseningError):
    """Partial decoding cannot preserve one-parent tree structure."""


class CompositionError(TreeCoarseningError):
    """Fitted stages cannot be combined into one lazy pipeline."""
