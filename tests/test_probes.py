"""Tests for the /failure-probes endpoint and its planning core."""
from __future__ import annotations

import random

from fastapi.testclient import TestClient

from app.main import app
from app.models import Node, ProbeObservation
from app.planner import PlanError
from app.probes import MAX_CLOSURE_NODES, plan_failure_probes

client = TestClient(app)


# ---------------------------------------------------------------- helpers


def node(name: str, deps: list[str] | None = None, content: str | None = None) -> Node:
    return Node(
        name=name, content_digest=content or name, dependency_names=deps or []
    )


def obs(name: str, result: str) -> ProbeObservation:
    return ProbeObservation(node=name, result=result)


def closure_set(graph: dict[str, list[str]], start: str) -> set[str]:
    seen: set[str] = set()
    stack = [start]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        stack.extend(graph[name])
    return seen


# ---------------------------------------------------------------- basics


def test_single_node_closure_needs_no_probes():
    result = plan_failure_probes([node("x")], "x", [])
    assert result.worst_case == 0
    assert result.tree.fault == "x"
    assert result.tree.probe is None


def test_chain_of_three():
    # a <- b <- t: probe a first; fail -> a, pass -> probe b (fail b / pass t).
    nodes = [node("a"), node("b", ["a"]), node("t", ["b"])]
    result = plan_failure_probes(nodes, "t", [])
    assert result.worst_case == 2
    tree = result.tree
    assert tree.probe == "a"
    assert tree.failed.fault == "a"
    assert tree.passed.probe == "b"
    assert tree.passed.failed.fault == "b"
    assert tree.passed.passed.fault == "t"


def test_chain_of_four_uses_balanced_prefix_probes():
    nodes = [
        node("a"),
        node("b", ["a"]),
        node("c", ["b"]),
        node("t", ["c"]),
    ]
    result = plan_failure_probes(nodes, "t", [])
    # Every probe on a chain offers a prefix split, so a balanced cut halving
    # the four candidates reaches ceil(log2 4) = 2 probes worst case.
    assert result.worst_case == 2


def test_chain_worst_depth_is_ceil_log2():
    for size in range(1, 11):
        nodes = [node("n0")] + [
            node(f"n{i}", [f"n{i-1}"]) for i in range(1, size)
        ]
        result = plan_failure_probes(nodes, f"n{size - 1}", [])
        expected = (size - 1).bit_length()
        assert result.worst_case == expected


def test_candidate_reduction_then_single_candidate_is_free():
    nodes = [node("a"), node("b", ["a"]), node("t", ["b"])]
    result = plan_failure_probes(nodes, "t", [obs("a", "fail")])
    assert result.worst_case == 0
    assert result.tree.fault == "a"

    result = plan_failure_probes(nodes, "t", [obs("a", "pass"), obs("b", "fail")])
    assert result.worst_case == 0
    assert result.tree.fault == "b"

    result = plan_failure_probes(nodes, "t", [obs("b", "pass")])
    assert result.worst_case == 0
    assert result.tree.fault == "t"


def test_observations_narrow_tree():
    nodes = [node("a"), node("b", ["a"]), node("t", ["b"])]
    result = plan_failure_probes(nodes, "t", [obs("a", "pass")])
    assert result.worst_case == 1
    assert result.tree.probe == "b"
    assert result.tree.passed.fault == "t"
    assert result.tree.failed.fault == "b"


def test_diamond_needs_three_without_outside_help():
    # base; b and c both depend on base; m on b,c; t on m.
    nodes = [
        node("base"),
        node("b", ["base"]),
        node("c", ["base"]),
        node("m", ["b", "c"]),
        node("t", ["m"]),
    ]
    result = plan_failure_probes(nodes, "t", [])
    assert result.worst_case == 3


def test_outside_node_with_non_principal_split_improves_worst_case():
    # A node outside the target closure may depend on several closure nodes,
    # inducing a fail-set that is not any closure node's principal closure.
    # x depending on the two independent leaves halves a diamond in one probe.
    nodes = [
        node("0"),
        node("1"),
        node("m", ["0", "1"]),
        node("t", ["m"]),
        node("x", ["0", "1"]),
    ]
    result = plan_failure_probes(nodes, "t", [])
    assert result.worst_case == 2
    assert result.tree.probe == "x"
    assert result.tree.failed.probe == "0"
    assert result.tree.passed.probe == "m"


def test_lexicographic_tie_break_among_equal_probes():
    nodes = [
        node("0"),
        node("1"),
        node("m", ["0", "1"]),
        node("t", ["m"]),
        node("a", ["0", "1"]),
        node("x", ["0", "1"]),
    ]
    result = plan_failure_probes(nodes, "t", [])
    assert result.worst_case == 2
    # "a" and "x" induce the exact same split; the smaller name wins.
    assert result.tree.probe == "a"


def test_tree_simulates_correctly_for_every_candidate():
    # Walk the returned tree for every possible faulty node; the reached
    # leaf must name that node, and the walked depth must not exceed
    # worst_case.
    nodes = [
        node("base"),
        node("b", ["base"]),
        node("c", ["base"]),
        node("m", ["b", "c"]),
        node("t", ["m"]),
        node("x", ["b"]),
    ]
    graph = {n.name: list(n.dependency_names) for n in nodes}
    result = plan_failure_probes(nodes, "t", [])
    closure = closure_set(graph, "t")

    def walk(fault: str):
        tree = result.tree
        depth = 0
        while tree.fault is None:
            fails = fault in closure_set(graph, tree.probe)
            tree = tree.failed if fails else tree.passed
            depth += 1
        return tree.fault, depth

    for fault in closure:
        leaf, depth = walk(fault)
        assert leaf == fault
        assert depth <= result.worst_case
    assert max(walk(f)[1] for f in closure) == result.worst_case


def test_probe_of_target_itself_is_never_planned():
    nodes = [node("a"), node("t", ["a"])]
    result = plan_failure_probes(nodes, "t", [])

    def check(tree) -> None:
        if tree.probe is not None:
            assert tree.probe != "t"
            check(tree.passed)
            check(tree.failed)

    check(result.tree)


def test_content_digest_does_not_matter():
    nodes_a = [node("a", content="aaa"), node("t", ["a"], content="ttt")]
    nodes_b = [node("a", content="zzz"), node("t", ["a"], content="qqq")]
    ra = plan_failure_probes(nodes_a, "t", [])
    rb = plan_failure_probes(nodes_b, "t", [])
    assert ra.model_dump() == rb.model_dump()


# ------------------------------------------------------------ observations


def test_observation_on_unrelated_outside_node_always_passes():
    nodes = [node("a"), node("b", ["a"]), node("t", ["b"]), node("q")]
    result = plan_failure_probes(nodes, "t", [obs("q", "pass")])
    assert result.worst_case == 2


def test_fail_observation_on_unrelated_outside_node_is_contradiction():
    nodes = [node("a"), node("b", ["a"]), node("t", ["b"]), node("q")]
    try:
        plan_failure_probes(nodes, "t", [obs("q", "fail")])
    except PlanError as exc:
        assert exc.code == "contradictory_observations"
    else:
        raise AssertionError("expected contradictory_observations")


def test_pass_observation_on_node_reaching_target_is_contradiction():
    # z depends on the target; probing z passing would clear the whole
    # closure, conflicting with the known target failure.
    nodes = [node("a"), node("t", ["a"]), node("z", ["t"])]
    try:
        plan_failure_probes(nodes, "t", [obs("z", "pass")])
    except PlanError as exc:
        assert exc.code == "contradictory_observations"
    else:
        raise AssertionError("expected contradictory_observations")


def test_pass_observation_on_target_is_contradiction():
    nodes = [node("a"), node("t", ["a"])]
    try:
        plan_failure_probes(nodes, "t", [obs("t", "pass")])
    except PlanError as exc:
        assert exc.code == "contradictory_observations"
    else:
        raise AssertionError("expected contradictory_observations")


def test_fail_observation_on_target_is_redundant_but_accepted():
    nodes = [node("a"), node("t", ["a"])]
    result = plan_failure_probes(nodes, "t", [obs("t", "fail")])
    assert result.worst_case == 1


def test_contradictory_pass_and_fail_pair():
    # Diamond: base; b and c both depend on base; t on b,c.  fail(c) leaves
    # {base,c,t}; pass(t) clears the whole closure -> impossible together.
    nodes = [
        node("base"),
        node("b", ["base"]),
        node("c", ["base"]),
        node("t", ["b", "c"]),
    ]
    try:
        plan_failure_probes(
            nodes, "t", [obs("c", "fail"), obs("t", "pass")]
        )
    except PlanError as exc:
        assert exc.code == "contradictory_observations"
    else:
        raise AssertionError("expected contradictory_observations")


def test_observation_order_does_not_matter():
    nodes = [
        node("base"),
        node("b", ["base"]),
        node("c", ["base"]),
        node("m", ["b", "c"]),
        node("t", ["m"]),
    ]
    r1 = plan_failure_probes(nodes, "t", [obs("c", "pass"), obs("b", "fail")])
    r2 = plan_failure_probes(nodes, "t", [obs("b", "fail"), obs("c", "pass")])
    assert r1.model_dump() == r2.model_dump()


# ------------------------------------------------------------- validation


def test_unknown_target():
    nodes = [node("a")]
    try:
        plan_failure_probes(nodes, "ghost", [])
    except PlanError as exc:
        assert exc.code == "unknown_target"
        assert exc.details == {"target": "ghost"}
    else:
        raise AssertionError("expected unknown_target")


def test_unknown_probe_node():
    nodes = [node("a"), node("t", ["a"])]
    try:
        plan_failure_probes(nodes, "t", [obs("ghost", "pass")])
    except PlanError as exc:
        assert exc.code == "unknown_probe_node"
        assert exc.details == {"nodes": ["ghost"]}
    else:
        raise AssertionError("expected unknown_probe_node")


def test_duplicate_observation_same_result_is_an_error():
    nodes = [node("a"), node("t", ["a"])]
    try:
        plan_failure_probes(nodes, "t", [obs("a", "pass"), obs("a", "pass")])
    except PlanError as exc:
        assert exc.code == "duplicate_observation"
        assert exc.details == {"nodes": ["a"]}
    else:
        raise AssertionError("expected duplicate_observation")


def test_duplicate_observation_conflicting_result_is_duplicate_not_contradiction():
    nodes = [node("a"), node("t", ["a"])]
    try:
        plan_failure_probes(nodes, "t", [obs("a", "pass"), obs("a", "fail")])
    except PlanError as exc:
        assert exc.code == "duplicate_observation"
    else:
        raise AssertionError("expected duplicate_observation")


def test_closure_limit_boundary():
    nodes = [node("n0")] + [
        node(f"n{i}", [f"n{i-1}"]) for i in range(1, MAX_CLOSURE_NODES)
    ]
    assert len(nodes) == MAX_CLOSURE_NODES
    result = plan_failure_probes(nodes, f"n{MAX_CLOSURE_NODES - 1}", [])
    assert result.worst_case == (MAX_CLOSURE_NODES - 1).bit_length()


def test_closure_too_large():
    nodes = [node("n0")] + [
        node(f"n{i}", [f"n{i-1}"]) for i in range(1, MAX_CLOSURE_NODES + 1)
    ]
    assert len(nodes) == MAX_CLOSURE_NODES + 1
    try:
        plan_failure_probes(nodes, f"n{MAX_CLOSURE_NODES}", [])
    except PlanError as exc:
        assert exc.code == "closure_too_large"
        assert exc.details == {
            "nodes": MAX_CLOSURE_NODES + 1,
            "limit": MAX_CLOSURE_NODES,
        }
    else:
        raise AssertionError("expected closure_too_large")


def test_graph_errors_propagate():
    # Missing dependency.
    try:
        plan_failure_probes([node("t", ["ghost"])], "t", [])
    except PlanError as exc:
        assert exc.code == "missing_dependency"
    else:
        raise AssertionError("expected missing_dependency")

    # Cycle.
    try:
        plan_failure_probes(
            [node("a", ["b"]), node("b", ["a"])], "a", []
        )
    except PlanError as exc:
        assert exc.code == "cycle_detected"
    else:
        raise AssertionError("expected cycle_detected")

    # Duplicate node.
    try:
        plan_failure_probes([node("a"), node("a")], "a", [])
    except PlanError as exc:
        assert exc.code == "duplicate_node"
    else:
        raise AssertionError("expected duplicate_node")


def test_payload_order_independence():
    nodes = [
        node("t", ["b", "c", "base"]),
        node("c", ["base"]),
        node("base"),
        node("b", ["base"]),
        node("x", ["c", "b"]),
    ]
    observations = [obs("c", "pass"), obs("x", "fail")]
    r1 = plan_failure_probes(nodes, "t", observations)

    shuffled = list(nodes)
    random.Random(7).shuffle(shuffled)
    r2 = plan_failure_probes(
        shuffled, "t", list(reversed(observations))
    )
    assert r1.model_dump() == r2.model_dump()


# ------------------------------------------------------- exhaustive oracle


def _reference_plan(graph: dict[str, list[str]], target: str, candidates: frozenset):
    """Independent minimax oracle over ALL graph nodes.

    Returns (worst_depth, chosen_probe_name) at the root; probes are all
    graph nodes and the name tie-break is applied independently per state.
    """
    closures = {name: closure_set(graph, name) for name in graph}

    def solve(state: frozenset) -> tuple[int, str | None]:
        if len(state) == 1:
            return 0, next(iter(state))
        best: int | None = None
        best_name: str | None = None
        for probe_name in sorted(graph):
            fail = state & closures[probe_name]
            passed = state - closures[probe_name]
            if not fail or not passed:
                continue
            value = 1 + max(solve(passed)[0], solve(fail)[0])
            if best is None or value < best:
                best = value
                best_name = probe_name
        assert best is not None
        return best, best_name

    return solve(candidates)


def test_reference_oracle_matches_on_random_graphs():
    rng = random.Random(20260928)
    for _trial in range(120):
        core_n = rng.randint(1, 7)
        names = [f"c{i}" for i in range(core_n)]
        graph: dict[str, list[str]] = {name: set() for name in names}
        for i in range(core_n):
            for j in range(i + 1, core_n):
                if rng.random() < 0.35:
                    graph[names[j]].add(names[i])
        # Ensure the target (last core node) transitively reaches all core
        # nodes so the closure is the whole core: add chain edges as needed.
        def transitive(start: str) -> set[str]:
            return closure_set({k: list(v) for k, v in graph.items()}, start)

        for i in range(core_n - 1):
            if names[i] not in transitive(names[-1]):
                graph[names[i + 1]].add(names[i])
        # Extra nodes outside the closure, with random dependencies on
        # core nodes (they create non-principal fail sets).
        extra_n = rng.randint(0, 4)
        for k in range(extra_n):
            ename = f"e{k}"
            deps = {
                names[i]
                for i in range(core_n)
                if rng.random() < 0.4
            }
            graph[ename] = deps
        graph_lists = {name: sorted(deps) for name, deps in graph.items()}
        nodes = [
            Node(name=name, content_digest=name, dependency_names=graph_lists[name])
            for name in sorted(graph)
        ]
        target = names[-1]

        # Random observations on random graph nodes.
        chosen_obs = []
        observed: set[str] = set()
        for _ in range(rng.randint(0, 3)):
            pname = rng.choice(sorted(graph))
            if pname in observed:
                continue
            observed.add(pname)
            chosen_obs.append(obs(pname, rng.choice(["pass", "fail"])))

        candidates = transitive(target)
        contradiction = False
        for ob in chosen_obs:
            if ob.result == "fail":
                candidates &= transitive(ob.node)
            else:
                candidates -= transitive(ob.node)
            if not candidates:
                contradiction = True
                break

        if contradiction:
            try:
                plan_failure_probes(nodes, target, chosen_obs)
            except PlanError as exc:
                assert exc.code == "contradictory_observations"
                continue
            raise AssertionError("expected contradictory_observations")

        result = plan_failure_probes(nodes, target, chosen_obs)
        ref_depth, ref_probe = _reference_plan(
            graph_lists, target, frozenset(candidates)
        )
        assert result.worst_case == ref_depth
        if len(candidates) > 1:
            assert result.tree.probe == ref_probe

        # Every candidate walks to its own leaf within worst_case probes.
        def walk(fault: str):
            tree = result.tree
            depth = 0
            while tree.fault is None:
                branch_fail = fault in transitive(tree.probe)
                tree = tree.failed if branch_fail else tree.passed
                depth += 1
            return tree.fault, depth

        for fault in candidates:
            leaf, depth = walk(fault)
            assert leaf == fault
            assert depth == ref_depth or depth < ref_depth
        assert max(walk(f)[1] for f in candidates) == ref_depth


# ----------------------------------------------------------------- HTTP


def _chain_payload():
    return {
        "nodes": [
            {"name": "a", "content_digest": "a", "dependency_names": []},
            {"name": "b", "content_digest": "b", "dependency_names": ["a"]},
            {"name": "t", "content_digest": "t", "dependency_names": ["b"]},
        ],
        "target": "t",
    }


def test_http_success_shape():
    response = client.post("/failure-probes", json=_chain_payload())
    assert response.status_code == 200
    body = response.json()
    assert body == {
        "worst_case": 2,
        "tree": {
            "probe": "a",
            "pass": {
                "probe": "b",
                "pass": {"fault": "t"},
                "fail": {"fault": "b"},
            },
            "fail": {"fault": "a"},
        },
    }
    # Leaves carry only `fault`; internal nodes never carry a null `fault`.
    assert set(body["tree"]["fail"]) == {"fault"}
    assert "fault" not in body["tree"]


def test_http_single_candidate_is_zero_probes():
    payload = _chain_payload()
    payload["observations"] = [{"node": "a", "result": "fail"}]
    response = client.post("/failure-probes", json=payload)
    assert response.status_code == 200
    assert response.json() == {"worst_case": 0, "tree": {"fault": "a"}}


def test_http_error_codes():
    payload = _chain_payload()
    payload["target"] = "ghost"
    response = client.post("/failure-probes", json=payload)
    assert response.status_code == 400
    assert response.json()["code"] == "unknown_target"

    payload = _chain_payload()
    payload["observations"] = [
        {"node": "a", "result": "pass"},
        {"node": "a", "result": "fail"},
    ]
    response = client.post("/failure-probes", json=payload)
    assert response.status_code == 400
    assert response.json()["code"] == "duplicate_observation"

    payload = _chain_payload()
    payload["observations"] = [{"node": "a", "result": "bogus"}]
    response = client.post("/failure-probes", json=payload)
    assert response.status_code == 422


def test_http_rejects_unknown_fields_and_bad_tree_inputs():
    payload = _chain_payload()
    payload["bogus"] = 1
    response = client.post("/failure-probes", json=payload)
    assert response.status_code == 422

    payload = _chain_payload()
    payload["target"] = ""
    response = client.post("/failure-probes", json=payload)
    assert response.status_code == 422

    payload = _chain_payload()
    payload["observations"] = [{"node": "", "result": "pass"}]
    response = client.post("/failure-probes", json=payload)
    assert response.status_code == 422


def test_observations_defaults_to_empty():
    response = client.post(
        "/failure-probes",
        json={
            "nodes": [
                {"name": "solo", "content_digest": "s", "dependency_names": []}
            ],
            "target": "solo",
        },
    )
    assert response.status_code == 200
    assert response.json() == {"worst_case": 0, "tree": {"fault": "solo"}}


def test_http_legacy_endpoints_still_work():
    response = client.get("/health")
    assert response.status_code == 200
    response = client.post(
        "/plans",
        json={
            "nodes": [
                {"name": "a", "content_digest": "a", "dependency_names": []}
            ]
        },
    )
    assert response.status_code == 201
