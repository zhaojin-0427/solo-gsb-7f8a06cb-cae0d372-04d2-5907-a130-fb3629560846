"""Tests for the /provenance/issue endpoint and its issuance core."""
from __future__ import annotations

import base64
import json
import random

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import Node
from app.planner import PlanError
from app.provenance import (
    attestation_digest,
    attestation_message,
    issue_provenance,
    verify_provenance,
)

client = TestClient(app)


# ---------------------------------------------------------------- helpers

SIGNER_1 = "builder-1"
SIGNER_2 = "builder-2"

# Fixed 32-byte seeds (same ones used by the verify golden vector).
SEED_1 = base64.b64decode("mWF1S/Qsd1Z1A2RE4tQFeDGofUyAaVdoAPQB6UBUnUA=")
SEED_2 = b"provenance-signer-two-seed-12345"[:32]
SEED_1_B64 = base64.b64encode(SEED_1).decode()
SEED_2_B64 = base64.b64encode(SEED_2).decode()
assert len(SEED_1) == len(SEED_2) == 32

GOLDEN_PUB = "kegzJbcb8pwgmbAFCH74+A1gvtw+MJImy+FxRK/RxAw="
GOLDEN_CORE_DIGEST = "35fda4461d08d70d865b13d5c63d4a1cba1bf5a00dabf651e42a89464e3ec4e6"
GOLDEN_CORE_SIG = (
    "FIuHzVQUY5GursLAKIinbwak0sDffPFx4D4GznlZx6BBMtPZkVYKquvEopQWSu39"
    "y6BPHtzBKSYE1nOATAWqBA=="
)
GOLDEN_UTIL_DIGEST = "ea88a79a0e9f526bfde053fb486642735c0b35e5bf30960ba91aed82892657a2"
GOLDEN_UTIL_SIG = (
    "RLMEy8ak2JPrxdUh7x0BXVU1mkaQiLpXJTeX+sWewMj60FQxwjFgU28AjWBPu8f4"
    "n7qr960/Wv8AgjRVOSxPDg=="
)


def node(name: str, content: str, deps: list[str] | None = None) -> Node:
    return Node(name=name, content_digest=content, dependency_names=deps or [])


def chain_nodes() -> list[Node]:
    # docs is deliberately outside util's closure.
    return [
        node("core", "core-src-v1"),
        node("util", "util-src-v1", ["core"]),
        node("docs", "docs-src-v1"),
    ]


def diamond_nodes() -> list[Node]:
    return [
        node("s", "s"),
        node("a", "a", ["s"]),
        node("b", "b", ["s"]),
        node("t", "t", ["a", "b"]),
        node("docs", "docs"),
    ]


def issue(
    *,
    nodes: list[Node] | None = None,
    target: str = "util",
    signers: dict[str, str] | None = None,
    private_keys: dict[str, str] | None = None,
):
    if signers is None:
        signers = {"core": SIGNER_1, "util": SIGNER_1}
    if private_keys is None:
        private_keys = {SIGNER_1: SEED_1_B64}
    return issue_provenance(
        nodes if nodes is not None else chain_nodes(),
        target,
        signers,
        private_keys,
    )


def verify_issued(nodes: list[Node], target: str, result) -> bool:
    """Feed an issuance result straight back into the verifier."""
    verdict = verify_provenance(
        nodes,
        target,
        list(result.attestations),
        dict(result.public_keys),
    )
    return verdict.trusted


# --------------------------------------------------------------- golden


def test_issued_chain_matches_golden_vector():
    result = issue()
    by_node = {a.node: a for a in result.attestations}
    assert list(by_node) == ["core", "util"]
    core, util = by_node["core"], by_node["util"]
    assert attestation_digest(core) == GOLDEN_CORE_DIGEST
    assert core.signature == GOLDEN_CORE_SIG
    assert attestation_digest(util) == GOLDEN_UTIL_DIGEST
    assert util.signature == GOLDEN_UTIL_SIG
    # util links the exact recomputed dependency digest.
    assert util.dependencies == {"core": GOLDEN_CORE_DIGEST}
    assert result.public_keys == {SIGNER_1: GOLDEN_PUB}


def test_attestation_digest_is_sha256_of_exact_canonical_bytes():
    result = issue()
    core = result.attestations[0]
    raw = json.dumps(
        [core.node, core.fingerprint, core.signer, []],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    assert attestation_message(core) == raw
    import hashlib

    assert hashlib.sha256(raw).hexdigest() == GOLDEN_CORE_DIGEST
    assert attestation_digest(core) == GOLDEN_CORE_DIGEST


def test_issued_chain_verifies_trusted():
    assert verify_issued(chain_nodes(), "util", issue()) is True


def test_two_signers_both_appear_in_public_table_and_verify():
    result = issue(
        signers={"core": SIGNER_1, "util": SIGNER_2},
        private_keys={SIGNER_1: SEED_1_B64, SIGNER_2: SEED_2_B64},
    )
    by_node = {a.node: a for a in result.attestations}
    assert by_node["core"].signer == SIGNER_1
    assert by_node["util"].signer == SIGNER_2
    assert set(result.public_keys) == {SIGNER_1, SIGNER_2}
    assert verify_issued(chain_nodes(), "util", result) is True


def test_closure_target_without_dependencies():
    result = issue(
        nodes=chain_nodes(),
        target="docs",
        signers={"docs": SIGNER_1},
        private_keys={SIGNER_1: SEED_1_B64},
    )
    assert [a.node for a in result.attestations] == ["docs"]
    assert result.attestations[0].dependencies == {}
    assert verify_issued(chain_nodes(), "docs", result) is True


def test_diamond_shared_dependency_is_attested_and_linked_once():
    signers = {name: SIGNER_1 for name in ("s", "a", "b", "t")}
    result = issue(nodes=diamond_nodes(), target="t", signers=signers)
    by_node = {a.node: a for a in result.attestations}
    assert [a.node for a in result.attestations] == ["a", "b", "s", "t"]
    s_digest = attestation_digest(by_node["s"])
    assert by_node["a"].dependencies == {"s": s_digest}
    assert by_node["b"].dependencies == {"s": s_digest}
    assert by_node["t"].dependencies == {
        "a": attestation_digest(by_node["a"]),
        "b": attestation_digest(by_node["b"]),
    }
    assert verify_issued(diamond_nodes(), "t", result) is True


def test_non_ascii_names_and_signers_are_not_escaped():
    nodes = [node("café", "é-content-ζ")]
    result = issue(
        nodes=nodes,
        target="café",
        signers={"café": "签署者α"},
        private_keys={"签署者α": SEED_1_B64},
    )
    att = result.attestations[0]
    message = attestation_message(att)
    assert b"caf\xc3\xa9" in message
    assert b"\xe7\xad\xbe\xe7\xbd\xb2\xe8\x80\x85\xce\xb1" in message
    assert b"\\u" not in message
    assert verify_issued(nodes, "café", result) is True


def test_input_order_does_not_change_output_bytes():
    baseline = json.dumps(issue().model_dump(), sort_keys=True)
    rng = random.Random(20260928)
    for _ in range(10):
        shuffled_nodes = chain_nodes()
        rng.shuffle(shuffled_nodes)
        shuffled_signers = {"core": SIGNER_1, "util": SIGNER_1}
        keys = list(shuffled_signers)
        rng.shuffle(keys)
        shuffled_signers = {name: shuffled_signers[name] for name in keys}
        result = issue(
            nodes=shuffled_nodes,
            signers=shuffled_signers,
            private_keys={SIGNER_1: SEED_1_B64},
        )
        assert json.dumps(result.model_dump(), sort_keys=True) == baseline


def test_attestations_are_sorted_by_node_name():
    nodes = [
        node("z", "z"),
        node("m", "m"),
        node("a", "a", ["z", "m"]),
    ]
    result = issue(
        nodes=nodes,
        target="a",
        signers={"z": SIGNER_1, "m": SIGNER_2, "a": SIGNER_1},
        private_keys={SIGNER_1: SEED_1_B64, SIGNER_2: SEED_2_B64},
    )
    assert [att.node for att in result.attestations] == ["a", "m", "z"]
    assert list(result.public_keys) == sorted(result.public_keys)
    assert verify_issued(nodes, "a", result) is True


# --------------------------------------------------------- hard errors


def test_duplicate_node_graph_error():
    with pytest.raises(PlanError) as exc:
        issue(
            nodes=[node("a", "a"), node("a", "b")],
            target="a",
            signers={"a": SIGNER_1},
            private_keys={SIGNER_1: SEED_1_B64},
        )
    assert exc.value.code == "duplicate_node"


def test_missing_dependency_graph_error():
    with pytest.raises(PlanError) as exc:
        issue(
            nodes=[node("a", "a", ["ghost"])],
            target="a",
            signers={"a": SIGNER_1},
            private_keys={SIGNER_1: SEED_1_B64},
        )
    assert exc.value.code == "missing_dependency"


def test_cycle_graph_error():
    with pytest.raises(PlanError) as exc:
        issue(
            nodes=[node("a", "a", ["b"]), node("b", "b", ["a"])],
            target="a",
            signers={"a": SIGNER_1, "b": SIGNER_1},
            private_keys={SIGNER_1: SEED_1_B64},
        )
    assert exc.value.code == "cycle_detected"


def test_unknown_target():
    with pytest.raises(PlanError) as exc:
        issue(target="ghost")
    assert exc.value.code == "unknown_target"
    assert exc.value.details == {"target": "ghost"}


def test_missing_signer_for_closure_node():
    with pytest.raises(PlanError) as exc:
        issue(signers={"util": SIGNER_1})
    assert exc.value.code == "missing_node_signer"
    assert exc.value.details == {"nodes": ["core"]}


def test_extra_signer_for_node_outside_closure():
    with pytest.raises(PlanError) as exc:
        issue(
            signers={"core": SIGNER_1, "util": SIGNER_1, "docs": SIGNER_1}
        )
    assert exc.value.code == "unexpected_node_signer"
    assert exc.value.details == {"nodes": ["docs"], "target": "util"}


def test_extra_signer_for_node_not_in_graph():
    with pytest.raises(PlanError) as exc:
        issue(
            signers={"core": SIGNER_1, "util": SIGNER_1, "phantom": SIGNER_1}
        )
    assert exc.value.code == "unexpected_node_signer"
    assert exc.value.details == {"nodes": ["phantom"], "target": "util"}


def test_missing_private_key_for_used_signer():
    with pytest.raises(PlanError) as exc:
        issue(
            signers={"core": SIGNER_1, "util": SIGNER_2},
            private_keys={SIGNER_1: SEED_1_B64},
        )
    assert exc.value.code == "missing_private_key"
    assert exc.value.details == {"signers": [SIGNER_2]}


def test_extra_unused_private_key_is_rejected():
    with pytest.raises(PlanError) as exc:
        issue(
            private_keys={SIGNER_1: SEED_1_B64, SIGNER_2: SEED_2_B64}
        )
    assert exc.value.code == "unexpected_private_key"
    assert exc.value.details == {"signers": [SIGNER_2]}


def test_unused_signer_key_covering_no_node_is_rejected_even_with_wrong_graph():
    # A signer used only by an out-of-closure node must not sneak its
    # private key past coverage checks (node map fails first).
    with pytest.raises(PlanError) as exc:
        issue(
            signers={"core": SIGNER_1, "util": SIGNER_1, "docs": SIGNER_2},
            private_keys={SIGNER_1: SEED_1_B64, SIGNER_2: SEED_2_B64},
        )
    assert exc.value.code == "unexpected_node_signer"


def test_malformed_private_key_base64():
    with pytest.raises(PlanError) as exc:
        issue(private_keys={SIGNER_1: "***"})
    assert exc.value.code == "malformed_encoding"
    assert exc.value.details == {"private_keys": [SIGNER_1]}


def test_malformed_private_key_length():
    with pytest.raises(PlanError) as exc:
        issue(private_keys={SIGNER_1: base64.b64encode(b"\x01" * 31).decode()})
    assert exc.value.code == "malformed_encoding"
    assert exc.value.details == {"private_keys": [SIGNER_1]}


def test_urlsafe_private_key_is_malformed():
    raw = bytes(range(32))
    value = base64.urlsafe_b64encode(raw).decode().rstrip("=")
    with pytest.raises(PlanError) as exc:
        issue(private_keys={SIGNER_1: value})
    assert exc.value.code == "malformed_encoding"


def test_multiple_malformed_private_keys_are_sorted():
    with pytest.raises(PlanError) as exc:
        issue(
            signers={"core": "z", "util": "a"},
            private_keys={"a": "***", "z": "bad"},
        )
    assert exc.value.code == "malformed_encoding"
    assert exc.value.details == {"private_keys": ["a", "z"]}


def test_unused_malformed_private_key_coverage_checked_before_decoding():
    # `m` is unused: the extra-key failure fires before keys are decoded, so a
    # malformed unused key value is never even read.
    with pytest.raises(PlanError) as exc:
        issue(
            signers={"core": "z", "util": "a"},
            private_keys={"a": "***", "z": "bad", "m": "also-bad"},
        )
    assert exc.value.code == "unexpected_private_key"
    assert exc.value.details == {"signers": ["m"]}


# ------------------------------------------------------------- no echo


def test_response_never_contains_private_material():
    import hashlib

    result = issue(
        signers={"core": SIGNER_1, "util": SIGNER_2},
        private_keys={SIGNER_1: SEED_1_B64, SIGNER_2: SEED_2_B64},
    )
    body = json.dumps(result.model_dump())
    assert SEED_1_B64 not in body
    assert SEED_2_B64 not in body
    for seed in (SEED_1, SEED_2):
        assert base64.b64encode(seed).decode() not in body
        assert seed.hex() not in body
        assert hashlib.sha256(seed).hexdigest() not in body


def test_error_messages_never_echo_private_material():
    cases = [
        dict(private_keys={SIGNER_1: "not-base64!!"}),
        dict(
            private_keys={SIGNER_1: SEED_1_B64, SIGNER_2: "not-base64!!"},
            signers={"core": SIGNER_1, "util": SIGNER_2},
        ),
    ]
    for overrides in cases:
        kwargs = {}
        kwargs.update(overrides)
        with pytest.raises(PlanError) as exc:
            issue(**kwargs)
        rendered = json.dumps(
            {"code": exc.value.code, "message": exc.value.message,
             "details": exc.value.details}
        )
        assert "not-base64" not in rendered
        assert SEED_1_B64 not in rendered


# --------------------------------------------------------------- HTTP


def _http_payload(**overrides):
    payload = {
        "nodes": [
            {
                "name": n.name,
                "content_digest": n.content_digest,
                "dependency_names": list(n.dependency_names),
            }
            for n in chain_nodes()
        ],
        "target": "util",
        "signers": {"core": SIGNER_1, "util": SIGNER_1},
        "private_keys": {SIGNER_1: SEED_1_B64},
    }
    payload.update(overrides)
    return payload


def test_http_issue_then_verify_roundtrip():
    response = client.post("/provenance/issue", json=_http_payload())
    assert response.status_code == 201
    issued = response.json()
    assert [a["node"] for a in issued["attestations"]] == ["core", "util"]
    assert issued["public_keys"] == {SIGNER_1: GOLDEN_PUB}
    assert SEED_1_B64 not in response.text

    verify_response = client.post(
        "/provenance/verify",
        json={
            "nodes": _http_payload()["nodes"],
            "target": "util",
            "attestations": issued["attestations"],
            "trusted_keys": issued["public_keys"],
        },
    )
    assert verify_response.status_code == 200
    assert verify_response.json() == {"trusted": True, "reasons": {}}


def test_http_two_signers_roundtrip():
    payload = _http_payload(
        signers={"core": SIGNER_1, "util": SIGNER_2},
        private_keys={SIGNER_1: SEED_1_B64, SIGNER_2: SEED_2_B64},
    )
    response = client.post("/provenance/issue", json=payload)
    assert response.status_code == 201
    issued = response.json()
    verify_response = client.post(
        "/provenance/verify",
        json={
            "nodes": payload["nodes"],
            "target": "util",
            "attestations": issued["attestations"],
            "trusted_keys": issued["public_keys"],
        },
    )
    assert verify_response.json() == {"trusted": True, "reasons": {}}


def test_http_unknown_target_is_400():
    response = client.post(
        "/provenance/issue", json=_http_payload(target="ghost")
    )
    assert response.status_code == 400
    assert response.json()["code"] == "unknown_target"


def test_http_missing_node_signer_is_400():
    response = client.post(
        "/provenance/issue",
        json=_http_payload(signers={"util": SIGNER_1}),
    )
    assert response.status_code == 400
    assert response.json()["code"] == "missing_node_signer"


def test_http_extra_node_signer_is_400():
    response = client.post(
        "/provenance/issue",
        json=_http_payload(
            signers={"core": SIGNER_1, "util": SIGNER_1, "docs": SIGNER_1}
        ),
    )
    assert response.status_code == 400
    assert response.json()["code"] == "unexpected_node_signer"


def test_http_missing_private_key_is_400():
    response = client.post(
        "/provenance/issue",
        json=_http_payload(
            signers={"core": SIGNER_1, "util": SIGNER_2},
            private_keys={SIGNER_1: SEED_1_B64},
        ),
    )
    assert response.status_code == 400
    assert response.json()["code"] == "missing_private_key"


def test_http_extra_private_key_is_400():
    response = client.post(
        "/provenance/issue",
        json=_http_payload(
            private_keys={SIGNER_1: SEED_1_B64, SIGNER_2: SEED_2_B64}
        ),
    )
    assert response.status_code == 400
    assert response.json()["code"] == "unexpected_private_key"


def test_http_malformed_private_key_is_400_and_does_not_echo():
    response = client.post(
        "/provenance/issue",
        json=_http_payload(private_keys={SIGNER_1: "%%%SECRET%%%"}),
    )
    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "malformed_encoding"
    assert body["details"] == {"private_keys": [SIGNER_1]}
    assert "SECRET" not in response.text


def test_http_non_string_private_key_value_is_400_and_does_not_echo():
    response = client.post(
        "/provenance/issue",
        json=_http_payload(private_keys={SIGNER_1: 1234567890}),
    )
    assert response.status_code == 400
    assert response.json()["code"] == "malformed_encoding"
    assert "1234567890" not in response.text


def test_http_graph_cycle_is_400():
    payload = _http_payload(
        nodes=[
            {"name": "a", "content_digest": "a", "dependency_names": ["b"]},
            {"name": "b", "content_digest": "b", "dependency_names": ["a"]},
        ],
        target="a",
        signers={"a": SIGNER_1, "b": SIGNER_1},
    )
    response = client.post("/provenance/issue", json=payload)
    assert response.status_code == 400
    assert response.json()["code"] == "cycle_detected"


def test_http_unknown_field_is_422():
    payload = _http_payload()
    payload["bogus"] = 1
    response = client.post("/provenance/issue", json=payload)
    assert response.status_code == 422
