"""Build-result provenance chain verification.

Given the current build graph (same schema as /plans), a release *target*
node, at most one signed attestation per node in the target's dependency
closure and a table of trusted Ed25519 public keys, verify that the whole
build-result provenance chain is authentic.

Attestations and their digests
------------------------------
Each :class:`~app.models.Attestation` names a node and claims:

* that node's recomputed ``fingerprint``;
* the ``signer`` id whose Ed25519 key signed the attestation;
* a mapping from every direct dependency name to that dependency's
  attestation digest.

The attestation's own *digest* is::

    digest = sha256(compact_json([
        node,
        fingerprint,
        signer,
        [[dep_name, dep_digest] for dep_name in sorted(dependencies)],
    ])).hexdigest()

where compact JSON is ``json.dumps(..., separators=(",", ":"),
ensure_ascii=False)`` encoded as UTF-8 (non-ASCII characters are never
escaped).  The Base64-decoded ``signature`` is verified as an Ed25519
signature over exactly those JSON bytes with the signer's trusted public
key.

Verification
------------
The graph fingerprints are recomputed with the /plans rules and the
target's transitive dependency closure is taken.  For every closure node
the service checks, independently, that:

* an attestation exists for the node (``missing_attestation``);
* the signer id is present in the trusted key table (``unknown_signer``);
* the claimed fingerprint equals the recomputed one
  (``fingerprint_mismatch``);
* the dependency mapping names exactly the node's direct dependencies
  (``dependency_mismatch`` for missing/extra names);
* each mapped dependency digest equals the dependency attestation's
  recomputed digest (``dependency_digest_mismatch``);
* the Ed25519 signature verifies (``invalid_signature``).

``trusted`` is true only when none of the above fires.  Failure reasons
are returned per node, sorted by node name; the reason codes inside a
node's list are sorted too, so the request payload order never matters.

Issuance
---------
The companion :func:`issue_provenance` powers ``POST /provenance/issue``.
Given the graph, the target, a node -> signer id assignment covering
exactly the target closure and a signer id -> Base64 raw 32-byte
Ed25519 private key seed table covering exactly the used signers, it
recomputes fingerprints in topological order and walks the closure in that
same order: each dependency attestation digest therefore already exists when the
parent is signed.  The produced attestations and the derived public-key table
are byte-for-byte acceptable to :func:`verify_provenance`, i.e.
resubmitting them yields ``trusted=true``.  Private seeds are used only to
sign; they are never echoed in the response or in any error message.

Hard errors (HTTP 400)
----------------------
Malformed graphs (duplicate node, missing dependency, cycle), unknown
targets, duplicate attestations for the same closure node, attestations
for nodes outside the closure (including nodes absent from the graph) and
malformed Base64 or wrong key/signature lengths produce structured
``PlanError`` responses instead of a trusted/untrusted verdict.  Issuance
additionally rejects signer assignments missing closure nodes or naming nodes
outside the closure, private-key tables missing used signers or carrying
unused entries, and malformed private-key seeds -- reporting node/signer ids
only, never the key material.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from app.models import (
    Attestation,
    Node,
    ProvenanceIssueResponse,
    ProvenanceVerifyResponse,
)
from app.planner import (
    PlanError,
    compute_fingerprints,
    validate_graph,
)

PUBLIC_KEY_BYTES = 32
PRIVATE_KEY_BYTES = 32
SIGNATURE_BYTES = 64


def _b64decode(value: str) -> bytes:
    """Strictly decode standard Base64; reject URL-safe or trailing junk."""
    return base64.b64decode(value, validate=True)


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


def attestation_message(attestation: Attestation) -> bytes:
    """Compact, non-ASCII-safe UTF-8 JSON bytes signed by the attestation."""
    pairs = sorted(attestation.dependencies.items())
    body = [
        attestation.node,
        attestation.fingerprint,
        attestation.signer,
        [[name, digest] for name, digest in pairs],
    ]
    return json.dumps(
        body,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def attestation_digest(attestation: Attestation) -> str:
    """Lowercase SHA-256 hex digest of the attestation's canonical message."""
    return hashlib.sha256(attestation_message(attestation)).hexdigest()


def _decode_trusted_keys(trusted_keys: dict[str, str]) -> dict[str, bytes]:
    """Base64-decode every trusted key; malformed encodings raise 400."""
    malformed: list[str] = []
    decoded: dict[str, bytes] = {}
    for signer_id in sorted(trusted_keys):
        try:
            raw_key = _b64decode(trusted_keys[signer_id])
        except (binascii.Error, ValueError):
            malformed.append(signer_id)
            continue
        if len(raw_key) != PUBLIC_KEY_BYTES:
            malformed.append(signer_id)
            continue
        decoded[signer_id] = raw_key
    if malformed:
        raise PlanError(
            "malformed_encoding",
            "Trusted public keys must be standard Base64 of a 32-byte "
            "Ed25519 public key",
            {"trusted_keys": malformed},
        )
    return decoded


def verify_provenance(
    nodes: list[Node],
    target: str,
    attestations: list[Attestation],
    trusted_keys: dict[str, str],
) -> ProvenanceVerifyResponse:
    """Verify the signed provenance chain over the target's dependency closure."""
    graph, contents, order = validate_graph(nodes)

    # ---- validation: target resolves -------------------------------------
    if target not in graph:
        raise PlanError(
            "unknown_target",
            f"Target node is not in the graph: {target!r}",
            {"target": target},
        )

    closure = _dependency_closure(graph, target)

    # ---- validation: attestations only name graph nodes ------------------
    unknown_nodes = sorted({a.node for a in attestations if a.node not in graph})
    if unknown_nodes:
        raise PlanError(
            "unknown_attestation_node",
            "Attestations reference nodes that are not in the graph",
            {"names": unknown_nodes},
        )

    # ---- validation: no attestations outside the target closure ----------
    outside = sorted({a.node for a in attestations if a.node not in closure})
    if outside:
        raise PlanError(
            "attestation_outside_closure",
            "Attestations may only cover nodes in the target's dependency closure",
            {"nodes": outside, "target": target},
        )

    # ---- validation: at most one attestation per closure node ------------
    seen_nodes: set[str] = set()
    duplicate_nodes: set[str] = set()
    for attestation in attestations:
        if attestation.node in seen_nodes:
            duplicate_nodes.add(attestation.node)
        seen_nodes.add(attestation.node)
    if duplicate_nodes:
        raise PlanError(
            "duplicate_attestation",
            "At most one attestation per closure node is allowed",
            {"nodes": sorted(duplicate_nodes)},
        )

    # ---- validation: Base64-encoded keys and signatures ------------------
    key_bytes = _decode_trusted_keys(trusted_keys)
    raw_signatures: dict[str, bytes] = {}
    malformed_signatures: list[str] = []
    for attestation in attestations:
        try:
            raw = _b64decode(attestation.signature)
        except (binascii.Error, ValueError):
            malformed_signatures.append(attestation.node)
            continue
        if len(raw) != SIGNATURE_BYTES:
            malformed_signatures.append(attestation.node)
            continue
        raw_signatures[attestation.node] = raw
    if malformed_signatures:
        raise PlanError(
            "malformed_encoding",
            "Attestation signatures must be standard Base64 of a 64-byte "
            "Ed25519 signature",
            {"signatures": sorted(malformed_signatures)},
        )

    # ---- recompute closure fingerprints ----------------------------------
    fingerprints = compute_fingerprints(graph, contents, order)
    by_node = {attestation.node: attestation for attestation in attestations}
    digests = {
        attestation.node: attestation_digest(attestation)
        for attestation in attestations
    }

    # ---- per-node checks --------------------------------------------------
    reasons: dict[str, list[str]] = {}
    for name in sorted(closure):
        node_reasons: set[str] = set()
        attestation = by_node.get(name)
        if attestation is None:
            reasons[name] = ["missing_attestation"]
            continue

        if attestation.fingerprint != fingerprints[name]:
            node_reasons.add("fingerprint_mismatch")

        claimed_deps = set(attestation.dependencies)
        actual_deps = set(graph[name])
        if claimed_deps != actual_deps:
            node_reasons.add("dependency_mismatch")
        for dep in sorted(claimed_deps & actual_deps):
            if attestation.dependencies[dep] != digests.get(dep):
                node_reasons.add("dependency_digest_mismatch")

        public_bytes = key_bytes.get(attestation.signer)
        if public_bytes is None:
            node_reasons.add("unknown_signer")
        else:
            try:
                Ed25519PublicKey.from_public_bytes(public_bytes).verify(
                    raw_signatures[name],
                    attestation_message(attestation),
                )
            except InvalidSignature:
                node_reasons.add("invalid_signature")

        if node_reasons:
            reasons[name] = sorted(node_reasons)

    return ProvenanceVerifyResponse(trusted=not reasons, reasons=reasons)


def _decode_private_keys(private_keys: dict) -> dict[str, Ed25519PrivateKey]:
    """Base64-decode every Ed25519 private seed; report malformed signer ids only.

    Key material itself is never placed in error messages or details.
    """
    malformed: list[str] = []
    decoded: dict[str, Ed25519PrivateKey] = {}
    for signer_id in sorted(private_keys, key=lambda v: str(v)):
        encoded = private_keys[signer_id]
        if not isinstance(encoded, str) or not encoded:
            malformed.append(signer_id)
            continue
        try:
            seed = _b64decode(encoded)
        except (binascii.Error, ValueError):
            malformed.append(signer_id)
            continue
        if len(seed) != PRIVATE_KEY_BYTES:
            malformed.append(signer_id)
            continue
        decoded[signer_id] = Ed25519PrivateKey.from_private_bytes(seed)
    if malformed:
        raise PlanError(
            "malformed_encoding",
            "Private keys must be standard Base64 of a 32-byte "
            "Ed25519 private key seed",
            {"private_keys": malformed},
        )
    return decoded


def issue_provenance(
    nodes: list[Node],
    target: str,
    signers: dict[str, str],
    private_keys: dict,
) -> ProvenanceIssueResponse:
    """Sign one provenance attestation per closure node; verify-ready output."""
    graph, contents, order = validate_graph(nodes)

    # ---- validation: target resolves -------------------------------------
    if target not in graph:
        raise PlanError(
            "unknown_target",
            f"Target node is not in the graph: {target!r}",
            {"target": target},
        )

    closure = _dependency_closure(graph, target)

    # ---- validation: node -> signer map covers the closure exactly ---------
    assigned = set(signers)
    missing_signers = sorted(closure - assigned)
    if missing_signers:
        raise PlanError(
            "missing_node_signer",
            "Every node in the target closure must name a signer",
            {"nodes": missing_signers},
        )
    extra_signers = sorted(assigned - closure)
    if extra_signers:
        raise PlanError(
            "unexpected_node_signer",
            "Signers may only be assigned to nodes in the target closure",
            {"nodes": extra_signers, "target": target},
        )

    # ---- validation: private-key table shape (keys are non-empty strings;
    # values are checked by the decoder; values are never echoed) -------------
    malformed_key_signers = sorted(
        signer_id
        for signer_id in private_keys
        if not isinstance(signer_id, str) or not signer_id.strip()
    )
    if malformed_key_signers:
        raise PlanError(
            "malformed_encoding",
            "Private-key signer ids must be non-empty strings",
            {"private_keys": malformed_key_signers},
        )

    # ---- validation: private-key table covers used signers exactly ----------
    used_signers = {signers[name] for name in closure}
    key_signers = set(private_keys)
    missing_keys = sorted(used_signers - key_signers)
    if missing_keys:
        raise PlanError(
            "missing_private_key",
            "Every used signer id must have a private key",
            {"signers": missing_keys},
        )
    extra_keys = sorted(key_signers - used_signers)
    if extra_keys:
        raise PlanError(
            "unexpected_private_key",
            "Private keys must cover exactly the signers used in the closure",
            {"signers": extra_keys},
        )

    # ---- validation: Base64-encoded 32-byte private seeds ---------------
    signing_keys = _decode_private_keys(private_keys)

    # ---- recompute fingerprints, then sign in topological order so each
    # dependency attestation digest exists before the parent is signed -----------
    fingerprints = compute_fingerprints(graph, contents, order)
    digests: dict[str, str] = {}
    by_node: dict[str, Attestation] = {}
    for name in order:
        if name not in closure:
            continue
        dependencies = {dep: digests[dep] for dep in graph[name]}
        unsigned = Attestation.model_construct(
            node=name,
            fingerprint=fingerprints[name],
            signer=signers[name],
            dependencies=dependencies,
            signature="",
        )
        message = attestation_message(unsigned)
        signature = signing_keys[signers[name]].sign(message)
        attestation = Attestation(
            node=name,
            fingerprint=fingerprints[name],
            signer=signers[name],
            dependencies=dependencies,
            signature=base64.b64encode(signature).decode("ascii"),
        )
        digests[name] = hashlib.sha256(message).hexdigest()
        by_node[name] = attestation

    public_keys = {
        signer_id: base64.b64encode(
            signing_keys[signer_id]
            .public_key()
            .public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        ).decode("ascii")
        for signer_id in sorted(used_signers)
    }
    return ProvenanceIssueResponse(
        attestations=[by_node[name] for name in sorted(closure)],
        public_keys=public_keys,
    )


__all__ = [
    "attestation_digest",
    "attestation_message",
    "issue_provenance",
    "verify_provenance",
]
