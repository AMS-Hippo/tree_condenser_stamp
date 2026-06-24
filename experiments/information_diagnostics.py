"""Experimental information diagnostics for ``tree_coarsening``.

This file deliberately lives outside the ``tree_coarsening`` package.  It uses
only the fitted public artifacts of :class:`tree_coarsening.EdgeBPECoarsener`
and ordinary NetworkX graphs.

The two main entry points are:

``analyze_edge_bpe_path``
    Replay a fitted edge-BPE rule list one rule at a time and measure structural
    compression and simple description-length diagnostics after every stage.

``TreeMarkovModel``
    Fit a smoothed parent-to-child label model on transformed training trees,
    then assign occurrence-level surprisal to vertices in a transformed held-out
    tree.  Both edge-weighted and parent-balanced transition estimates are
    available.  A sibling-predictive score uses the corresponding
    Dirichlet-multinomial posterior predictive within each child group.

These are exploratory diagnostics, not a universal compressor.  In particular,
the dictionary and topology terms are explicit, inspectable surrogates rather
than claims of an optimal code.

References used by the companion notebook include Shannon (1948), Benjamini and
Peres (1994, DOI 10.1214/aop/1176988857), Krichevsky and Trofimov (1981, DOI
10.1109/TIT.1981.1056331), and Rissanen (1978, DOI
10.1016/0005-1098(78)90005-5).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from math import ceil, isfinite, lgamma, log, log2
from typing import Any, Literal, TypeAlias, cast

import networkx as nx

from tree_coarsening import (
    EdgeBPECoarsener,
    EdgeBPEEncoder,
    EdgeBPERule,
    MatchingLabel,
)

Token: TypeAlias = MatchingLabel

TransitionWeighting: TypeAlias = Literal["edge", "parent"]
TopologyCode: TypeAlias = Literal["none", "catalan", "balanced_parentheses"]

# A private category shared by every label unseen during ``fit``.  It is never
# written to a graph or exposed in a result record.
_UNKNOWN_CATEGORY = object()


@dataclass(frozen=True, slots=True)
class VertexInformation:
    """Information assigned to one vertex occurrence.

    ``transition_surprisal_bits`` is a first-order tree-Markov score.  For a
    non-root vertex it is ``-log2 q(label | parent_label)``; for a root it uses
    the fitted root-label distribution.

    ``sibling_surprisal_bits`` uses an exchangeable Dirichlet-multinomial
    posterior predictive within the current parent's child group.  The total
    score for a child bag is order invariant, although individual contributions
    are reported in deterministic timestamp order.

    Rule fields describe the *token type* learned during BPE fitting.  Context
    fields describe this particular occurrence in the held-out tree.  They are
    intentionally kept separate rather than summed automatically.
    """

    node: Hashable
    label: Token
    parent: Hashable | None
    parent_label: Token | None
    is_root: bool
    size: int
    transition_probability: float
    transition_surprisal_bits: float
    transition_surprisal_per_site_bits: float
    transition_association_bits: float | None
    sibling_probability: float | None
    sibling_surprisal_bits: float | None
    sibling_surprisal_per_site_bits: float | None
    rule_local_probability: float | None
    rule_local_surprisal_bits: float | None
    rule_local_association_bits: float | None
    rule_construction_surprisal_bits: float | None
    rule_construction_surprisal_per_site_bits: float | None

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GraphCodeSummary:
    """Corpus-level code and size statistics at one replay stage."""

    n_graphs: int
    n_nodes: int
    n_edges: int
    n_roots: int
    represented_sites: int
    transition_bits: float
    sibling_predictive_bits: float
    topology_bits: float

    @property
    def transition_bits_per_site(self) -> float:
        return self.transition_bits / self.represented_sites

    @property
    def sibling_predictive_bits_per_site(self) -> float:
        return self.sibling_predictive_bits / self.represented_sites

    @property
    def topology_bits_per_site(self) -> float:
        return self.topology_bits / self.represented_sites

    def to_record(self, prefix: str = "") -> dict[str, Any]:
        return {
            f"{prefix}n_graphs": self.n_graphs,
            f"{prefix}n_nodes": self.n_nodes,
            f"{prefix}n_edges": self.n_edges,
            f"{prefix}n_roots": self.n_roots,
            f"{prefix}represented_sites": self.represented_sites,
            f"{prefix}transition_bits": self.transition_bits,
            f"{prefix}sibling_predictive_bits": self.sibling_predictive_bits,
            f"{prefix}topology_bits": self.topology_bits,
            f"{prefix}transition_bits_per_site": self.transition_bits_per_site,
            f"{prefix}sibling_predictive_bits_per_site": (
                self.sibling_predictive_bits_per_site
            ),
            f"{prefix}topology_bits_per_site": self.topology_bits_per_site,
        }


@dataclass(frozen=True, slots=True)
class RuleInformation:
    """Information attached to one learned edge-BPE token type."""

    rank: int
    token: Token
    parent_label: Token
    child_label: Token
    raw_count: int
    fit_actual_events: int
    replay_train_events: int
    replay_validation_events: int | None
    transition_weighting: TransitionWeighting
    mle_transition_probability: float | None
    smoothed_transition_probability: float
    child_base_probability: float
    local_surprisal_bits: float
    local_association_bits: float
    construction_surprisal_bits: float
    construction_association_bits: float
    represented_sites: int
    construction_surprisal_per_site_bits: float

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StageInformation:
    """Structural and coding diagnostics after ``stage`` fitted rules."""

    stage: int
    last_rule_token: Token | None
    dictionary_bits: float
    train: GraphCodeSummary
    validation: GraphCodeSummary | None
    train_compression_rate: float
    validation_compression_rate: float | None
    train_transition_total_bits: float
    train_sibling_total_bits: float
    validation_transition_total_bits: float | None
    validation_sibling_total_bits: float | None
    train_transition_gain_bits: float | None
    train_sibling_gain_bits: float | None
    validation_transition_gain_bits: float | None
    validation_sibling_gain_bits: float | None
    train_nodes_removed: int | None
    validation_nodes_removed: int | None

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "stage": self.stage,
            "last_rule_token": self.last_rule_token,
            "dictionary_bits": self.dictionary_bits,
            "train_compression_rate": self.train_compression_rate,
            "validation_compression_rate": self.validation_compression_rate,
            "train_transition_total_bits": self.train_transition_total_bits,
            "train_sibling_total_bits": self.train_sibling_total_bits,
            "validation_transition_total_bits": self.validation_transition_total_bits,
            "validation_sibling_total_bits": self.validation_sibling_total_bits,
            "train_transition_gain_bits": self.train_transition_gain_bits,
            "train_sibling_gain_bits": self.train_sibling_gain_bits,
            "validation_transition_gain_bits": self.validation_transition_gain_bits,
            "validation_sibling_gain_bits": self.validation_sibling_gain_bits,
            "train_nodes_removed": self.train_nodes_removed,
            "validation_nodes_removed": self.validation_nodes_removed,
            "train_transition_gain_per_removed_node": _safe_rate(
                self.train_transition_gain_bits, self.train_nodes_removed
            ),
            "train_sibling_gain_per_removed_node": _safe_rate(
                self.train_sibling_gain_bits, self.train_nodes_removed
            ),
            "validation_transition_gain_per_removed_node": _safe_rate(
                self.validation_transition_gain_bits, self.validation_nodes_removed
            ),
            "validation_sibling_gain_per_removed_node": _safe_rate(
                self.validation_sibling_gain_bits, self.validation_nodes_removed
            ),
        }
        record.update(self.train.to_record("train_"))
        if self.validation is not None:
            record.update(self.validation.to_record("validation_"))
        return record


@dataclass(frozen=True, slots=True)
class InformationPath:
    """Complete output of :func:`analyze_edge_bpe_path`."""

    stages: tuple[StageInformation, ...]
    rules: tuple[RuleInformation, ...]
    transition_weighting: TransitionWeighting
    alpha: float
    base_pseudocount: float
    topology_code: TopologyCode

    @property
    def rule_by_token(self) -> dict[Token, RuleInformation]:
        return {rule.token: rule for rule in self.rules}

    def stage_records(self) -> list[dict[str, Any]]:
        return [stage.to_record() for stage in self.stages]

    def rule_records(self) -> list[dict[str, Any]]:
        return [rule.to_record() for rule in self.rules]


@dataclass(slots=True)
class TreeMarkovModel:
    """Smoothed first-order model for labels on directed rooted trees.

    The basic factorization is

    ``p(root labels) * product_(u,v in E) p(label(v) | label(u))``.

    ``weighting='edge'`` samples a directed edge uniformly when estimating the
    transition table.  ``weighting='parent'`` first samples a non-leaf parent
    uniformly and then samples one of its children uniformly.  The latter stops
    one extremely high-degree parent from dominating the fitted distribution.

    The posterior-predictive transition probability is

    ``q(b | a) = (n_ab + alpha * p0(b)) / (n_a + alpha)``,

    where ``p0`` is a smoothed global child-label distribution.  This is a
    finite hierarchical Dirichlet smoothing rule with a transparent strength
    parameter ``alpha``.

    Labels absent from the fitted corpus are all scored through one reserved
    unknown-label bucket.  This keeps the smoothed probabilities normalized:
    several distinct unseen Python objects do not each receive a full copy of
    the unknown mass.
    """

    alpha: float = 8.0
    base_pseudocount: float = 0.5
    weighting: TransitionWeighting = "edge"
    label_attr: str = "label"
    size_attr: str = "size"
    time_attr: str = "time"

    vertex_counts_: dict[Token, float] = field(init=False, default_factory=dict)
    root_counts_: dict[Token, float] = field(init=False, default_factory=dict)
    child_counts_: dict[Token, float] = field(init=False, default_factory=dict)
    transition_counts_: dict[Token, dict[Token, float]] = field(
        init=False, default_factory=dict
    )
    context_totals_: dict[Token, float] = field(init=False, default_factory=dict)
    vocabulary_: frozenset[Token] = field(init=False, default_factory=frozenset)
    total_vertices_: float = field(init=False, default=0.0)
    total_roots_: float = field(init=False, default=0.0)
    total_transition_weight_: float = field(init=False, default=0.0)
    fitted_: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if not isfinite(self.alpha) or self.alpha <= 0.0:
            raise ValueError("alpha must be a finite positive number.")
        if not isfinite(self.base_pseudocount) or self.base_pseudocount <= 0.0:
            raise ValueError("base_pseudocount must be a finite positive number.")
        if self.weighting not in {"edge", "parent"}:
            raise ValueError("weighting must be 'edge' or 'parent'.")

    def fit(self, graphs: nx.DiGraph | Sequence[nx.DiGraph]) -> "TreeMarkovModel":
        graph_list = _as_graph_list(graphs)

        vertex_counts: defaultdict[Token, float] = defaultdict(float)
        root_counts: defaultdict[Token, float] = defaultdict(float)
        child_counts: defaultdict[Token, float] = defaultdict(float)
        transition_counts: defaultdict[Token, defaultdict[Token, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        context_totals: defaultdict[Token, float] = defaultdict(float)
        vocabulary: set[Token] = set()

        for graph in graph_list:
            _validate_directed_tree(graph)
            for _node, data in graph.nodes(data=True):
                label = cast(Token, data[self.label_attr])
                vocabulary.add(label)
                vertex_counts[label] += 1.0

            roots = [node for node in graph if graph.in_degree(node) == 0]
            if not roots:
                raise ValueError("every graph must contain at least one directed root.")
            for root in roots:
                root_label = cast(Token, graph.nodes[root][self.label_attr])
                root_counts[root_label] += 1.0

            for parent in graph:
                children = list(graph.successors(parent))
                if not children:
                    continue
                parent_label = cast(Token, graph.nodes[parent][self.label_attr])
                edge_weight = 1.0 if self.weighting == "edge" else 1.0 / len(children)
                for child in children:
                    child_label = cast(Token, graph.nodes[child][self.label_attr])
                    transition_counts[parent_label][child_label] += edge_weight
                    context_totals[parent_label] += edge_weight
                    child_counts[child_label] += edge_weight

        self.vertex_counts_ = dict(vertex_counts)
        self.root_counts_ = dict(root_counts)
        self.child_counts_ = dict(child_counts)
        self.transition_counts_ = {
            parent: dict(counts) for parent, counts in transition_counts.items()
        }
        self.context_totals_ = dict(context_totals)
        self.vocabulary_ = frozenset(vocabulary)
        self.total_vertices_ = float(sum(vertex_counts.values()))
        self.total_roots_ = float(sum(root_counts.values()))
        self.total_transition_weight_ = float(sum(child_counts.values()))
        self.fitted_ = True
        return self

    def _require_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError("Call fit before requesting probabilities or scores.")

    def _category(self, label: Token) -> Token | object:
        return label if label in self.vocabulary_ else _UNKNOWN_CATEGORY

    @property
    def alphabet_size_with_unknown(self) -> int:
        self._require_fitted()
        return len(self.vocabulary_) + 1

    def unigram_probability(self, label: Token) -> float:
        self._require_fitted()
        denominator = (
            self.total_vertices_
            + self.base_pseudocount * self.alphabet_size_with_unknown
        )
        category = self._category(label)
        return (
            self.vertex_counts_.get(cast(Token, category), 0.0)
            + self.base_pseudocount
        ) / denominator

    def child_base_probability(self, label: Token) -> float:
        self._require_fitted()
        denominator = (
            self.total_transition_weight_
            + self.base_pseudocount * self.alphabet_size_with_unknown
        )
        category = self._category(label)
        return (
            self.child_counts_.get(cast(Token, category), 0.0)
            + self.base_pseudocount
        ) / denominator

    def root_probability(self, label: Token) -> float:
        self._require_fitted()
        category = self._category(label)
        base = self.unigram_probability(label)
        return (
            self.root_counts_.get(cast(Token, category), 0.0) + self.alpha * base
        ) / (self.total_roots_ + self.alpha)

    def transition_probability(self, parent_label: Token, child_label: Token) -> float:
        self._require_fitted()
        parent_category = cast(Token, self._category(parent_label))
        child_category = cast(Token, self._category(child_label))
        count = self.transition_counts_.get(parent_category, {}).get(
            child_category, 0.0
        )
        total = self.context_totals_.get(parent_category, 0.0)
        base = self.child_base_probability(child_label)
        return (count + self.alpha * base) / (total + self.alpha)

    def transition_mle(
        self, parent_label: Token, child_label: Token
    ) -> float | None:
        self._require_fitted()
        parent_category = cast(Token, self._category(parent_label))
        child_category = cast(Token, self._category(child_label))
        total = self.context_totals_.get(parent_category, 0.0)
        if total <= 0.0:
            return None
        return (
            self.transition_counts_.get(parent_category, {}).get(child_category, 0.0)
            / total
        )

    def transition_association_bits(
        self, parent_label: Token, child_label: Token
    ) -> float:
        """Return ``log2 q(child|parent) / p0(child)``.

        Positive values mean that the child is more likely in this parent
        context than under the global child-label distribution.  This is an
        association contrast, not a self-information value.
        """

        conditional = self.transition_probability(parent_label, child_label)
        base = self.child_base_probability(child_label)
        return log2(conditional / base)

    def score_vertices(
        self,
        graph: nx.DiGraph,
        *,
        rule_information: Mapping[Token, RuleInformation] | None = None,
    ) -> tuple[VertexInformation, ...]:
        self._require_fitted()
        _validate_directed_tree(graph)
        parent_of: dict[Hashable, Hashable] = {}
        for node in graph:
            predecessors = list(graph.predecessors(node))
            if len(predecessors) > 1:
                raise ValueError(
                    f"vertex {node!r} has indegree {len(predecessors)}; expected a tree."
                )
            if predecessors:
                parent_of[node] = predecessors[0]

        sibling_probabilities: dict[Hashable, float] = {}
        for parent in graph:
            children = sorted(
                graph.successors(parent),
                key=lambda child: _node_order_key(
                    graph,
                    child,
                    label_attr=self.label_attr,
                    time_attr=self.time_attr,
                ),
            )
            if not children:
                continue
            parent_label = cast(Token, graph.nodes[parent][self.label_attr])
            parent_category = cast(Token, self._category(parent_label))
            prior_total = self.context_totals_.get(parent_category, 0.0) + self.alpha
            seen: Counter[Token | object] = Counter()
            for position, child in enumerate(children):
                child_label = cast(Token, graph.nodes[child][self.label_attr])
                child_category = self._category(child_label)
                prior_count = self.transition_counts_.get(parent_category, {}).get(
                    cast(Token, child_category), 0.0
                ) + self.alpha * self.child_base_probability(child_label)
                probability = (prior_count + seen[child_category]) / (
                    prior_total + position
                )
                sibling_probabilities[child] = probability
                seen[child_category] += 1

        records: list[VertexInformation] = []
        for node in graph:
            data = graph.nodes[node]
            label = cast(Token, data[self.label_attr])
            size = int(data.get(self.size_attr, 1))
            parent = parent_of.get(node)
            if parent is None:
                probability = self.root_probability(label)
                parent_label = None
                association = None
                sibling_probability = None
                sibling_surprisal = None
            else:
                parent_label = cast(Token, graph.nodes[parent][self.label_attr])
                probability = self.transition_probability(parent_label, label)
                association = self.transition_association_bits(parent_label, label)
                sibling_probability = sibling_probabilities[node]
                sibling_surprisal = -log2(sibling_probability)

            rule = None if rule_information is None else rule_information.get(label)
            records.append(
                VertexInformation(
                    node=node,
                    label=label,
                    parent=parent,
                    parent_label=parent_label,
                    is_root=parent is None,
                    size=size,
                    transition_probability=probability,
                    transition_surprisal_bits=-log2(probability),
                    transition_surprisal_per_site_bits=(
                        -log2(probability) / size
                    ),
                    transition_association_bits=association,
                    sibling_probability=sibling_probability,
                    sibling_surprisal_bits=sibling_surprisal,
                    sibling_surprisal_per_site_bits=(
                        None if sibling_surprisal is None else sibling_surprisal / size
                    ),
                    rule_local_probability=(
                        None if rule is None else rule.smoothed_transition_probability
                    ),
                    rule_local_surprisal_bits=(
                        None if rule is None else rule.local_surprisal_bits
                    ),
                    rule_local_association_bits=(
                        None if rule is None else rule.local_association_bits
                    ),
                    rule_construction_surprisal_bits=(
                        None if rule is None else rule.construction_surprisal_bits
                    ),
                    rule_construction_surprisal_per_site_bits=(
                        None
                        if rule is None
                        else rule.construction_surprisal_per_site_bits
                    ),
                )
            )
        return tuple(records)

    def annotate_graph(
        self,
        graph: nx.DiGraph,
        *,
        rule_information: Mapping[Token, RuleInformation] | None = None,
        prefix: str = "info_",
        copy: bool = True,
    ) -> nx.DiGraph:
        """Return ``graph`` with per-vertex information attributes added."""

        output = graph.copy(as_view=False) if copy else graph
        for record in self.score_vertices(output, rule_information=rule_information):
            data = output.nodes[record.node]
            data[f"{prefix}transition_probability"] = record.transition_probability
            data[f"{prefix}transition_surprisal_bits"] = (
                record.transition_surprisal_bits
            )
            data[f"{prefix}transition_surprisal_per_site_bits"] = (
                record.transition_surprisal_per_site_bits
            )
            data[f"{prefix}transition_association_bits"] = (
                record.transition_association_bits
            )
            data[f"{prefix}sibling_probability"] = record.sibling_probability
            data[f"{prefix}sibling_surprisal_bits"] = record.sibling_surprisal_bits
            data[f"{prefix}sibling_surprisal_per_site_bits"] = (
                record.sibling_surprisal_per_site_bits
            )
            data[f"{prefix}rule_local_probability"] = record.rule_local_probability
            data[f"{prefix}rule_local_surprisal_bits"] = (
                record.rule_local_surprisal_bits
            )
            data[f"{prefix}rule_local_association_bits"] = (
                record.rule_local_association_bits
            )
            data[f"{prefix}rule_construction_surprisal_bits"] = (
                record.rule_construction_surprisal_bits
            )
            data[f"{prefix}rule_construction_surprisal_per_site_bits"] = (
                record.rule_construction_surprisal_per_site_bits
            )
        return output


def ordered_tree_shape_bits(n_nodes: int) -> float:
    """Ideal code length ``log2 Catalan(n_nodes - 1)`` for an ordered tree shape."""

    if n_nodes < 1:
        raise ValueError("n_nodes must be positive.")
    if n_nodes == 1:
        return 0.0
    k = n_nodes - 1
    log_catalan = lgamma(2 * k + 1) - 2.0 * lgamma(k + 1) - log(k + 1)
    return log_catalan / log(2.0)


def topology_code_bits(graph: nx.DiGraph, mode: TopologyCode = "catalan") -> float:
    """Return an explicit topology-code surrogate for one tree."""

    n_nodes = graph.number_of_nodes()
    if n_nodes < 1:
        raise ValueError("graphs must be nonempty.")
    if mode == "none":
        return 0.0
    if mode == "catalan":
        return ordered_tree_shape_bits(n_nodes)
    if mode == "balanced_parentheses":
        return float(2 * n_nodes)
    raise ValueError("mode must be 'none', 'catalan', or 'balanced_parentheses'.")


def summarize_corpus_code(
    model: TreeMarkovModel,
    graphs: nx.DiGraph | Sequence[nx.DiGraph],
    *,
    topology: TopologyCode = "catalan",
) -> GraphCodeSummary:
    """Score one graph or graph sequence under a fitted model."""

    graph_list = _as_graph_list(graphs)
    n_nodes = 0
    n_edges = 0
    n_roots = 0
    represented_sites = 0
    transition_bits = 0.0
    sibling_bits = 0.0
    topology_bits = 0.0

    for graph in graph_list:
        records = model.score_vertices(graph)
        n_nodes += graph.number_of_nodes()
        n_edges += graph.number_of_edges()
        topology_bits += topology_code_bits(graph, topology)
        for record in records:
            represented_sites += record.size
            transition_bits += record.transition_surprisal_bits
            if record.is_root:
                n_roots += 1
                sibling_bits += record.transition_surprisal_bits
            else:
                assert record.sibling_surprisal_bits is not None
                sibling_bits += record.sibling_surprisal_bits

    if represented_sites <= 0:
        raise ValueError("represented site count must be positive.")
    return GraphCodeSummary(
        n_graphs=len(graph_list),
        n_nodes=n_nodes,
        n_edges=n_edges,
        n_roots=n_roots,
        represented_sites=represented_sites,
        transition_bits=transition_bits,
        sibling_predictive_bits=sibling_bits,
        topology_bits=topology_bits,
    )


def edge_bpe_dictionary_bits(
    coarsener: EdgeBPECoarsener,
    n_rules: int,
    *,
    operation_overhead_bits: float = 0.0,
    integer_width: bool = False,
) -> float:
    """Return a transparent prefix-rule-table code-length surrogate.

    At rank ``r``, both endpoints are references into the ``K0 + r`` symbols
    available before the new token is created.  The new token is implied by
    rank.  Thus the idealized rule cost is

    ``2 log2(K0 + r) + operation_overhead_bits``.

    Set ``integer_width=True`` to use two fixed-width integer references,
    ``2 ceil(log2(K0 + r))``, instead of the idealized real-valued length.
    The base vocabulary cost is intentionally omitted because it is constant
    across prefixes of one fitted model.
    """

    encoder = _require_edge_encoder(coarsener)
    if n_rules < 0 or n_rules > len(encoder.edge_rules):
        raise ValueError("n_rules lies outside the fitted rule prefix.")
    if not isfinite(operation_overhead_bits) or operation_overhead_bits < 0.0:
        raise ValueError("operation_overhead_bits must be finite and nonnegative.")

    base_size = len(encoder.base_labels)
    total = 0.0
    for rank in range(n_rules):
        choices = max(1, base_size + rank)
        reference_bits = float(ceil(log2(choices))) if integer_width else log2(choices)
        total += 2.0 * reference_bits + operation_overhead_bits
    return total


def analyze_edge_bpe_path(
    coarsener: EdgeBPECoarsener,
    train_graphs: nx.DiGraph | Sequence[nx.DiGraph],
    *,
    validation_graphs: nx.DiGraph | Sequence[nx.DiGraph] | None = None,
    alpha: float = 8.0,
    base_pseudocount: float = 0.5,
    transition_weighting: TransitionWeighting = "edge",
    topology: TopologyCode = "catalan",
    operation_overhead_bits: float = 0.0,
    integer_dictionary_width: bool = False,
    validate_replay: bool = False,
    max_rules: int | None = None,
) -> InformationPath:
    """Replay a fitted edge-BPE model one rule at a time.

    This function does not refit or inspect private training state.  It takes the
    fitted ``EdgeBPEEncoder`` artifact, makes one-rule encoder views with
    :func:`dataclasses.replace`, and applies them sequentially to copies of the
    supplied trees.

    A model is fitted on the current training representation before each merge.
    Consequently, the returned rule surprisal is stage-specific: a nested token
    is judged using the label distribution that existed when that rule became
    eligible.  Held-out validation code lengths are preferable to in-sample
    lengths for choosing a prefix.
    """

    encoder = _require_edge_encoder(coarsener)
    rules = encoder.edge_rules
    if max_rules is not None:
        if max_rules < 0:
            raise ValueError("max_rules must be nonnegative or None.")
        if max_rules > len(rules):
            raise ValueError("max_rules exceeds the number of fitted edge-BPE rules.")
        rules = rules[:max_rules]

    train_current = [graph.copy(as_view=False) for graph in _as_graph_list(train_graphs)]
    validation_current = (
        None
        if validation_graphs is None
        else [
            graph.copy(as_view=False)
            for graph in _as_graph_list(validation_graphs)
        ]
    )

    initial_train_nodes = sum(graph.number_of_nodes() for graph in train_current)
    initial_validation_nodes = (
        None
        if validation_current is None
        else sum(graph.number_of_nodes() for graph in validation_current)
    )

    stages: list[StageInformation] = []
    rule_rows: list[RuleInformation] = []
    rule_by_token: dict[Token, RuleInformation] = {}
    previous_train: GraphCodeSummary | None = None
    previous_validation: GraphCodeSummary | None = None
    previous_train_transition_total: float | None = None
    previous_train_sibling_total: float | None = None
    previous_validation_transition_total: float | None = None
    previous_validation_sibling_total: float | None = None

    for stage in range(len(rules) + 1):
        model = TreeMarkovModel(
            alpha=alpha,
            base_pseudocount=base_pseudocount,
            weighting=transition_weighting,
            label_attr=coarsener.label_attr,
            size_attr=coarsener.size_attr,
            time_attr=coarsener.time_attr,
        ).fit(train_current)

        train_summary = summarize_corpus_code(model, train_current, topology=topology)
        validation_summary = (
            None
            if validation_current is None
            else summarize_corpus_code(model, validation_current, topology=topology)
        )
        dictionary_bits = edge_bpe_dictionary_bits(
            coarsener,
            stage,
            operation_overhead_bits=operation_overhead_bits,
            integer_width=integer_dictionary_width,
        )
        train_transition_total = (
            train_summary.transition_bits + train_summary.topology_bits + dictionary_bits
        )
        train_sibling_total = (
            train_summary.sibling_predictive_bits
            + train_summary.topology_bits
            + dictionary_bits
        )
        validation_transition_total = (
            None
            if validation_summary is None
            else validation_summary.transition_bits
            + validation_summary.topology_bits
            + dictionary_bits
        )
        validation_sibling_total = (
            None
            if validation_summary is None
            else validation_summary.sibling_predictive_bits
            + validation_summary.topology_bits
            + dictionary_bits
        )

        stages.append(
            StageInformation(
                stage=stage,
                last_rule_token=None if stage == 0 else rules[stage - 1].token,
                dictionary_bits=dictionary_bits,
                train=train_summary,
                validation=validation_summary,
                train_compression_rate=(
                    1.0 - train_summary.n_nodes / initial_train_nodes
                ),
                validation_compression_rate=(
                    None
                    if validation_summary is None or initial_validation_nodes is None
                    else 1.0
                    - validation_summary.n_nodes / initial_validation_nodes
                ),
                train_transition_total_bits=train_transition_total,
                train_sibling_total_bits=train_sibling_total,
                validation_transition_total_bits=validation_transition_total,
                validation_sibling_total_bits=validation_sibling_total,
                train_transition_gain_bits=(
                    None
                    if previous_train_transition_total is None
                    else previous_train_transition_total - train_transition_total
                ),
                train_sibling_gain_bits=(
                    None
                    if previous_train_sibling_total is None
                    else previous_train_sibling_total - train_sibling_total
                ),
                validation_transition_gain_bits=(
                    None
                    if previous_validation_transition_total is None
                    or validation_transition_total is None
                    else previous_validation_transition_total
                    - validation_transition_total
                ),
                validation_sibling_gain_bits=(
                    None
                    if previous_validation_sibling_total is None
                    or validation_sibling_total is None
                    else previous_validation_sibling_total - validation_sibling_total
                ),
                train_nodes_removed=(
                    None
                    if previous_train is None
                    else previous_train.n_nodes - train_summary.n_nodes
                ),
                validation_nodes_removed=(
                    None
                    if previous_validation is None or validation_summary is None
                    else previous_validation.n_nodes - validation_summary.n_nodes
                ),
            )
        )

        previous_train = train_summary
        previous_validation = validation_summary
        previous_train_transition_total = train_transition_total
        previous_train_sibling_total = train_sibling_total
        previous_validation_transition_total = validation_transition_total
        previous_validation_sibling_total = validation_sibling_total

        if stage == len(rules):
            break

        rule = rules[stage]
        history = _history_row(coarsener, stage)
        smoothed_probability = model.transition_probability(
            rule.parent_label, rule.child_label
        )
        child_base_probability = model.child_base_probability(rule.child_label)
        local_surprisal = -log2(smoothed_probability)
        local_association = log2(smoothed_probability / child_base_probability)
        parent_info = rule_by_token.get(rule.parent_label)
        child_info = rule_by_token.get(rule.child_label)
        construction_surprisal = local_surprisal
        construction_association = local_association
        if parent_info is not None:
            construction_surprisal += parent_info.construction_surprisal_bits
            construction_association += parent_info.construction_association_bits
        if child_info is not None:
            construction_surprisal += child_info.construction_surprisal_bits
            construction_association += child_info.construction_association_bits

        train_nodes_before = sum(graph.number_of_nodes() for graph in train_current)
        validation_nodes_before = (
            None
            if validation_current is None
            else sum(graph.number_of_nodes() for graph in validation_current)
        )
        one_rule_encoder = _one_rule_encoder(encoder, stage)
        train_current = [
            one_rule_encoder.encode(graph, validate=validate_replay)
            for graph in train_current
        ]
        if validation_current is not None:
            validation_current = [
                one_rule_encoder.encode(graph, validate=validate_replay)
                for graph in validation_current
            ]
        train_nodes_after = sum(graph.number_of_nodes() for graph in train_current)
        validation_nodes_after = (
            None
            if validation_current is None
            else sum(graph.number_of_nodes() for graph in validation_current)
        )
        replay_train_events = train_nodes_before - train_nodes_after
        if validate_replay and "actual_events" in history:
            fitted_events = int(history["actual_events"])
            if replay_train_events != fitted_events:
                raise RuntimeError(
                    "one-rule replay disagrees with the fitted event count at "
                    f"rank {stage}: replay={replay_train_events}, "
                    f"fit={fitted_events}."
                )

        represented_sites = encoder.vocab.site_count(rule.token)
        rule_info = RuleInformation(
            rank=rule.rank,
            token=rule.token,
            parent_label=rule.parent_label,
            child_label=rule.child_label,
            raw_count=rule.count,
            fit_actual_events=int(history.get("actual_events", 0)),
            replay_train_events=replay_train_events,
            replay_validation_events=(
                None
                if validation_nodes_before is None or validation_nodes_after is None
                else validation_nodes_before - validation_nodes_after
            ),
            transition_weighting=transition_weighting,
            mle_transition_probability=model.transition_mle(
                rule.parent_label, rule.child_label
            ),
            smoothed_transition_probability=smoothed_probability,
            child_base_probability=child_base_probability,
            local_surprisal_bits=local_surprisal,
            local_association_bits=local_association,
            construction_surprisal_bits=construction_surprisal,
            construction_association_bits=construction_association,
            represented_sites=represented_sites,
            construction_surprisal_per_site_bits=(
                construction_surprisal / represented_sites
            ),
        )
        rule_rows.append(rule_info)
        rule_by_token[rule.token] = rule_info

    return InformationPath(
        stages=tuple(stages),
        rules=tuple(rule_rows),
        transition_weighting=transition_weighting,
        alpha=alpha,
        base_pseudocount=base_pseudocount,
        topology_code=topology,
    )


def fit_transformed_markov_model(
    coarsener: EdgeBPECoarsener,
    train_graphs: nx.DiGraph | Sequence[nx.DiGraph],
    *,
    alpha: float = 8.0,
    base_pseudocount: float = 0.5,
    weighting: TransitionWeighting = "edge",
    validate_transform: bool = False,
) -> tuple[TreeMarkovModel, list[nx.DiGraph]]:
    """Transform the training corpus and fit a final-stage information model."""

    _require_edge_encoder(coarsener)
    transformed = coarsener.transform(train_graphs, validate=validate_transform)
    transformed_list = _as_graph_list(transformed)
    model = TreeMarkovModel(
        alpha=alpha,
        base_pseudocount=base_pseudocount,
        weighting=weighting,
        label_attr=coarsener.label_attr,
        size_attr=coarsener.size_attr,
        time_attr=coarsener.time_attr,
    ).fit(transformed_list)
    return model, transformed_list


def _require_edge_encoder(coarsener: EdgeBPECoarsener) -> EdgeBPEEncoder:
    encoder = coarsener.encoder_
    if encoder is None:
        raise RuntimeError("The EdgeBPECoarsener must be fitted first.")
    if not isinstance(encoder, EdgeBPEEncoder):
        raise TypeError(
            "Expected an EdgeBPEEncoder artifact; this experiment currently "
            "supports EdgeBPECoarsener only."
        )
    if len(encoder.edge_rules) != len(encoder.rules):
        raise RuntimeError("edge rule metadata and encoder rule metadata disagree.")
    return encoder


def _one_rule_encoder(encoder: EdgeBPEEncoder, rank: int) -> EdgeBPEEncoder:
    if rank < 0 or rank >= len(encoder.edge_rules):
        raise IndexError(rank)
    return replace(
        encoder,
        edge_rules=(encoder.edge_rules[rank],),
        rules=(encoder.rules[rank],),
    )


def _history_row(coarsener: EdgeBPECoarsener, rank: int) -> Mapping[str, Any]:
    if rank < len(coarsener.history_):
        row = coarsener.history_[rank]
        if int(row.get("rank", rank)) != rank:
            raise RuntimeError("coarsener history is not aligned with fitted rule order.")
        return row
    return {}


def _as_graph_list(
    graphs: nx.DiGraph | Sequence[nx.DiGraph],
) -> list[nx.DiGraph]:
    if isinstance(graphs, nx.DiGraph):
        return [graphs]
    if isinstance(graphs, (str, bytes, bytearray)):
        raise TypeError("graphs must be a DiGraph or a sequence of DiGraphs.")
    output = list(graphs)
    if not output:
        raise ValueError("at least one graph is required.")
    if not all(isinstance(graph, nx.DiGraph) for graph in output):
        raise TypeError("every graph must be a networkx.DiGraph.")
    return output


def _safe_rate(numerator: float | None, denominator: int | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _validate_directed_tree(graph: nx.DiGraph) -> Hashable:
    if not isinstance(graph, nx.DiGraph) or graph.is_multigraph():
        raise TypeError("each graph must be a networkx.DiGraph, not a multigraph.")
    n_nodes = graph.number_of_nodes()
    if n_nodes < 1:
        raise ValueError("trees must be nonempty.")
    if graph.number_of_edges() != n_nodes - 1:
        raise ValueError(
            f"a tree with {n_nodes} nodes must have {n_nodes - 1} edges."
        )
    roots = [node for node in graph if graph.in_degree(node) == 0]
    if len(roots) != 1:
        raise ValueError(f"expected exactly one directed root; found {len(roots)}.")
    root = roots[0]
    bad = [
        node
        for node in graph
        if node != root and graph.in_degree(node) != 1
    ]
    if bad:
        raise ValueError(
            "every non-root vertex must have in-degree one; "
            f"bad vertices include {bad[:5]!r}."
        )
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("the directed tree must be acyclic.")
    return root


def _node_order_key(
    graph: nx.DiGraph,
    node: Hashable,
    *,
    label_attr: str,
    time_attr: str,
) -> tuple[float, str, str]:
    data = graph.nodes[node]
    return (
        float(data.get(time_attr, 0.0)),
        repr(data[label_attr]),
        repr(node),
    )


__all__ = [
    "GraphCodeSummary",
    "InformationPath",
    "RuleInformation",
    "StageInformation",
    "TopologyCode",
    "TransitionWeighting",
    "TreeMarkovModel",
    "VertexInformation",
    "analyze_edge_bpe_path",
    "edge_bpe_dictionary_bits",
    "fit_transformed_markov_model",
    "ordered_tree_shape_bits",
    "summarize_corpus_code",
    "topology_code_bits",
]
