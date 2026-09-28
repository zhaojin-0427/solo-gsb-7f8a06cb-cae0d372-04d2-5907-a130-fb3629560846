"""Single-fault build probe planning.

Given the current build graph (same schema as /plans), a *target* node whose
build is known to have failed, and optional already-completed probe
observations, plan an adaptive decision tree that pinpoints the unique faulty
node with the minimum possible worst-case number of additional probes.

Fault and probe model
---------------------
Exactly one node in the target's dependency closure (the target itself plus
every transitive dependency) is faulty — the closure is assumed to hold at
most ``MAX_CLOSURE_NODES`` (10) nodes.  Probing a node *succeeds* when the
fault is neither that node nor any of its transitive dependencies, and
*fails* otherwise::

    probe(n) fails  <=>  fault in closure(n)

so the set of candidate fault nodes surviving a *pass* outcome at *n* is
``candidates - closure(n)``, while a *fail* outcome leaves
``candidates & closure(n)``.

The target's own failed build is itself a "fail" observation at the target,
hence the target itself can never be ruled out: the starting candidate set is
the whole closure.

Optimal tree
------------
Every node of the *whole* graph is a legal probe — not just closure nodes.  A
node outside the closure whose dependencies include several closure nodes
induces a fail-set equal to the union of their closures; such a union need not
be the closure of any single closure node, so probing it can split the
candidates in a way no closure node offers and reduce the worst-case depth.
A node wholly disconnected from the closure has an empty fail-set and can never
be optimal.  The minimax number of additional probes is computed exactly over
candidate subsets of the (at most ten) closure nodes::

    cost({x})        = 0
    cost(S)          = min over probe nodes p, both non-empty branches,
                       of 1 + max(cost(S - closure(p)), cost(S & closure(p)))

The optimal probe at each state is the smallest-name graph node attaining that
minimum (nodes inducing the same fail-set are equivalent, so one representative
per distinct set suffices); ties at different states are resolved
independently, per branch.  The same DP builds the complete adaptive tree.

Determinism
-----------
Inputs are validated and processed through name-sorted structures at every
step; the order of the supplied observations never changes the result.
"""
from __future__ import annotations

from app.models import FailureProbeResponse, Node, ProbeObservation, ProbeTreeNode
from app.planner import PlanError, validate_graph

MAX_CLOSURE_NODES = 10


def _dependency_closure(graph: dict[str, list[str]], target: str) -> set[str]:
    """Target and every node reachable through direct dependencies."""
    closure: set[str] = set()
    stack = [target]
    while stack:
        name = stack.pop()
        if name in closure:
            continue
        closure.add(name)
        stack.extend(graph[name])
    return closure


def plan_failure_probes(
    nodes: list[Node],
    target: str,
    observations: list[ProbeObservation],
) -> FailureProbeResponse:
    """Narrow candidates by observations and build the optimal probe tree."""
    graph, _contents, order = validate_graph(nodes)

    # ---- validation: observations are unique probes ----------------------
    seen_nodes: set[str] = set()
    duplicate_nodes: set[str] = set()
    for observation in observations:
        if observation.node in seen_nodes:
            duplicate_nodes.add(observation.node)
        seen_nodes.add(observation.node)
    if duplicate_nodes:
        raise PlanError(
            "duplicate_observation",
            "Observation node names must be unique",
            {"nodes": sorted(duplicate_nodes)},
        )

    # ---- validation: every observation references a graph node -----------
    unknown_probes = sorted(
        {observation.node for observation in observations} - set(graph)
    )
    if unknown_probes:
        raise PlanError(
            "unknown_probe_node",
            "Observations reference nodes that are not in the graph",
            {"nodes": unknown_probes},
        )

    # ---- validation: target resolves -------------------------------------
    if target not in graph:
        raise PlanError(
            "unknown_target",
            f"Target node is not in the graph: {target!r}",
            {"target": target},
        )

    # ---- validation: closure size limit -----------------------------------
    closure = _dependency_closure(graph, target)
    if len(closure) > MAX_CLOSURE_NODES:
        raise PlanError(
            "closure_too_large",
            f"The target closure has {len(closure)} nodes; at most "
            f"{MAX_CLOSURE_NODES} are supported",
            {"nodes": len(closure), "limit": MAX_CLOSURE_NODES},
        )
    closure_sorted = sorted(closure)
    index = {name: i for i, name in enumerate(closure_sorted)}

    # ---- closure bitmask for every graph node, restricted to the closure --
    # Computed in topological order so a node's mask is itself ORed with the
    # already-computed masks of its direct dependencies.
    closure_mask: dict[str, int] = {}
    for name in order:
        mask = 1 << index[name] if name in closure else 0
        for dep in graph[name]:
            mask |= closure_mask[dep]
        closure_mask[name] = mask

    # ---- narrow the candidate set using the observations -----------------
    # probe(n) fails <=> fault in closure(n).  The target's failed build
    # already establishes fail(target), so the starting set is the full
    # closure.
    candidates = (1 << len(closure_sorted)) - 1
    for observation in sorted(observations, key=lambda item: item.node):
        mask = closure_mask[observation.node]
        if observation.result == "fail":
            candidates &= mask
        else:
            candidates &= ~mask

    if candidates == 0:
        raise PlanError(
            "contradictory_observations",
            "No single faulty node in the target closure is consistent with "
            "all observations",
            {"target": target},
        )

    # ---- distinct closure masks, each by the lexicographically smallest ---
    # node that induces it: the probe at any state is chosen by name among
    # all equally optimal graph nodes, so only one representative per mask
    # can ever win.
    probe_name: dict[int, str] = {}
    for name in sorted(graph):
        mask = closure_mask[name]
        if mask not in probe_name:
            probe_name[mask] = name
    # Representatives ordered by node name so the first minimizer scanned
    # wins the lexicographic tie-break.
    probe_masks = sorted(probe_name, key=probe_name.get)

    # ---- exact minimax DP + tree construction over candidate subsets ------
    memo: dict[int, tuple[int, ProbeTreeNode]] = {}

    def solve(mask: int) -> tuple[int, ProbeTreeNode]:
        if mask in memo:
            return memo[mask]
        if mask & (mask - 1) == 0:
            leaf = ProbeTreeNode(fault=closure_sorted[mask.bit_length() - 1])
            memo[mask] = (0, leaf)
            return memo[mask]

        best: tuple[int, int] | None = None  # (worst-case probes, probe mask)
        for probe_mask in probe_masks:
            fail_mask = mask & probe_mask
            pass_mask = mask & ~probe_mask
            if fail_mask == 0 or pass_mask == 0:
                continue  # branch with no possible fault: uninformative
            probes = 1 + max(solve(pass_mask)[0], solve(fail_mask)[0])
            if best is None or probes < best[0]:
                best = (probes, probe_mask)

        # A multi-candidate state always offers an informative probe:
        # probing any minimal (dependency-free within the state) candidate
        # splits that singleton off from the remaining candidates.
        assert best is not None
        probes, probe_mask = best
        subtree = ProbeTreeNode(
            probe=probe_name[probe_mask],
            passed=solve(mask & ~probe_mask)[1],
            failed=solve(mask & probe_mask)[1],
        )
        memo[mask] = (probes, subtree)
        return memo[mask]

    worst_case, tree = solve(candidates)
    return FailureProbeResponse(worst_case=worst_case, tree=tree)


__all__ = ["MAX_CLOSURE_NODES", "plan_failure_probes"]
