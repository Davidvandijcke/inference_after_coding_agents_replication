#!/usr/bin/env python3
"""Bounded declaration export and provenance-checked label integration; no judging.

Only indexed, hash-verified closed records are eligible for public packets.
This module never extracts archives to disk, loads credentials, reads SQLite,
executes scientific programs, or computes scientific estimates.
"""
import argparse
import hashlib
import hmac
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import tarfile
import tempfile

FIELDS = ("model", "null", "population", "outcome", "sample_restrictions")
SCHEMA = "card-fixed3000-review-input-v1"
LIMIT = 1024 * 1024


def need(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else canonical(value)).hexdigest()


def controller_commitment_digest(value):
    """Match the frozen controller's declaration hash, including ASCII escapes.

    This domain is only the five-field commitment. Public-pair, index and file
    hashes retain their existing canonicalization and exact-byte checks.
    """
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()


def reference(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), "Regular non-symlink input required")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(LIMIT), b""):
            h.update(block)
    return {"path": str(path.resolve()), "sha256": h.hexdigest()}


def checked(ref):
    need(set(ref) == {"path", "sha256"} and reference(ref["path"]) ==
         {"path": str(Path(ref["path"]).resolve()), "sha256": ref["sha256"]},
         "Input reference mismatch")
    return Path(ref["path"])


def read(ref):
    return json.loads(checked(ref).read_bytes())


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    raw = canonical(value) + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    return {"path": str(path.resolve()), "sha256": sha(raw)}


def declaration(value):
    return (isinstance(value, dict) and set(value) == set(FIELDS) and
            all(isinstance(value[k], str) and value[k].strip() for k in FIELDS))


def roster_rows(roster):
    rows = roster["selected_fresh_jobs"]
    need(roster["schema"] == "card3000-materialized-roster-v1" and
         roster["fixed_quotas"] == {"tail": 1142, "body": 1858} and len(rows) == 3000,
         "Exact frozen 3000 roster required")
    need(len({r["job_id"] for r in rows}) == 3000 and
         len({r["world_id"] for r in rows}) == 3000, "Unique job and world IDs required")
    need(sum(r["stratum"] == "tail" for r in rows) == 1142 and
         sum(r["stratum"] == "body" for r in rows) == 1858, "Frozen quotas differ")
    need([r["stratum"] for r in rows] == ["tail"] * 1142 + ["body"] * 1858 and
         all(r["shard"] == i % 4 and r["order_position"] == i+1 for i, r in enumerate(rows)),
         "Frozen round-robin roster order differs")
    return rows


def index_records(pack):
    """Hash the small immutable index snapshot; never hash the complete pack."""
    path = checked(pack["index"])
    records = []; prefix = hashlib.sha256(); end = 0; seen = set(); terminal = None; header_seen = False
    with path.open("rb") as f:
        for number, line in enumerate(f):
            need(len(line) <= 4096 and line.endswith(b"\n"), "Only complete bounded index lines allowed")
            row = json.loads(line); seal = row.pop("sha256", None)
            need(seal == sha(row) and row.get("schema") == "scientific-case-pack-v1",
                 "Index self seal differs")
            if number == 0:
                need(row["kind"] == "header" and row["source_binding_sha256"] ==
                     pack["source_binding_sha256"], "Checkpoint source binding differs")
                header_seen = True
            elif row["kind"] == "case":
                need(terminal is None and row["case_id"] not in seen and
                     type(row["offset"]) is int and type(row["length"]) is int and
                     row["offset"] == end and row["length"] > 0,
                     "Duplicate, overlapping or unordered indexed record")
                end += row["length"]; seen.add(row["case_id"])
                records.append(dict(row, index_record_sha256=seal))
            elif row["kind"] == "terminal":
                need(terminal is None and row["case_count"] == len(records) and
                     row["index_prefix_sha256"] == prefix.hexdigest(), "Terminal index seal differs")
                terminal = row
            else:
                raise ValueError("Unknown committed index kind")
            prefix.update(line)
    need(header_seen, "Missing checkpoint header")
    need(end <= pack["committed_pack_bytes"], "Index exceeds exported committed prefix")
    pack_path = Path(pack["path"])
    need(pack_path.is_file() and not pack_path.is_symlink() and pack_path.stat().st_size >= end,
         "Exported pack prefix truncated")
    if pack.get("terminal_required", False):
        need(terminal is not None and terminal["uncommitted_pack_bytes"] == 0,
             "Sealed checkpoint required")
    return records


def closed_result(pack, record):
    """One range read into a temporary spool, SHA verified before tar parsing."""
    h = hashlib.sha256()
    with tempfile.TemporaryFile() as spool:
        with Path(pack["path"]).open("rb") as f:
            f.seek(record["offset"]); remaining = record["length"]
            while remaining:
                block = f.read(min(LIMIT, remaining))
                need(bool(block), "Closed range truncated")
                spool.write(block); h.update(block); remaining -= len(block)
        need(h.hexdigest() == record["payload_sha256"], "Closed payload digest differs")
        spool.seek(0); result = None
        with tarfile.open(fileobj=spool, mode="r|") as archive:
            for member in archive:
                name = PurePosixPath(member.name)
                need(not name.is_absolute() and ".." not in name.parts,
                     "Unsafe archive member name")
                if str(name) == "case/result.json":
                    need(result is None and member.isfile() and not member.issym() and
                         not member.islnk() and member.size <= LIMIT,
                         "Duplicate or unsafe closed result")
                    result = archive.extractfile(member).read()
        need(result is not None, "Closed record has no exact case/result.json")
        return json.loads(result), sha(result)


def release_hold(value):
    # Numeric model parameters are declarations; embedded realized results need
    # a separately versioned public redaction, never a silent semantic rewrite.
    text = "\n".join(value.values())
    return bool(re.search(r"(?i)(?:\bp[- _]?value\s*[:=<>]|\bp\s*[=<>]\s*0[.]|"
                          r"\b(?:estimate|standard error|t.statistic|z.statistic)\s*[:=]|"
                          r"\bBearer\s+\S+|\bsk-[A-Za-z0-9_-]{12,})", text))


def export(binding_ref, destination):
    binding = read(binding_ref)
    need(binding["schema"] == SCHEMA, "Wrong extraction binding")
    rows = roster_rows(read(binding["roster"])); original = read(binding["original"])
    target = original["commitment"]
    need(original["status"] == "committed" and declaration(target) and
         original["commitment_hash"] == controller_commitment_digest(target), "Committed target digest differs")
    need(not release_hold(target), "Target declaration needs versioned public blinding")
    contexts = []
    for ref in binding["contexts"]:
        raw = checked(ref).read_bytes()
        need(len(raw) <= LIMIT, "Oversized public context")
        contexts.append({"name": Path(ref["path"]).name, "sha256": sha(raw), "text": raw.decode()})
    context_version = sha([{k: r[k] for k in ("name", "sha256")} for r in contexts])
    previous = read(binding["previous_private"]) if binding.get("previous_private") else None
    salt = previous["blinding_salt"] if previous else secrets.token_hex(32)
    if previous:
        need(previous["roster"] == binding["roster"] and previous["original"] == binding["original"] and
             previous["contexts"] == binding["contexts"], "Incremental review authority changed")
    by_id = {r["job_id"]: r for r in rows}
    old = {r["job_id"]: r for r in previous["rows"]} if previous else {}
    observations = {k: dict(v) for k, v in old.items()}
    closed_seen = {k: v["closed_record_sha256"] for k, v in old.items()
                   if v.get("closed_record_sha256")}
    claimed = set(previous.get("claimed_ids", [])) if previous else set()
    key_to_id = {sha(j.encode())[:24]: j for j in by_id}
    new_packets = []; read_bytes = 0
    for pack in binding["packs"]:
        need(type(pack["shard"]) is int and 0 <= pack["shard"] < 4, "Wrong pack shard")
        # Executor owns export; this receipt supplies stable-prefix provenance.
        receipt = read(pack["export_receipt"])
        need(receipt["schema"] == "card-fixed3000-checkpoint-prefix-export-v1" and
             receipt["index_sha256"] == pack["index"]["sha256"] and
             receipt["shard"] == pack["shard"] and
             receipt["committed_pack_bytes"] == pack["committed_pack_bytes"],
             "Executor export receipt differs")
        for record in index_records(pack):
            key, sep, phase = record["case_id"].rpartition("-")
            need(sep and key in key_to_id and phase in {"claim", "closed"}, "Unexpected checkpoint case ID")
            job_id = key_to_id[key]; position = next(i for i, r in enumerate(rows) if r["job_id"] == job_id)
            need(position % 4 == pack["shard"], "Wrong frozen case shard")
            if phase == "claim":
                claimed.add(job_id); continue
            if job_id in closed_seen:
                need(closed_seen[job_id] == record["payload_sha256"], "Previously exported closure changed")
                continue
            result, result_sha = closed_result(pack, record); read_bytes += record["length"]
            need(result.get("job_id") == job_id and result.get("shard") == pack["shard"] and
                 result.get("status") in {"complete", "unavailable"}, "Unknown closed result identity/status")
            raw = result.get("self_verdict")
            raw_match = raw.get("model_null_match") if isinstance(raw, dict) else None
            raw_valid = (result["status"] == "complete" and type(raw_match) is bool and
                         isinstance(raw.get("reason"), str) and bool(raw["reason"].strip()))
            candidate = result.get("declared_analysis")
            declaration_ok = declaration(candidate)
            if result["status"] == "complete":
                need(declaration_ok and controller_commitment_digest(candidate) == result.get("commitment_hash") and
                     result.get("target_commitment_hash") == controller_commitment_digest(target), "Complete declaration lineage differs")
            blind = hmac.new(bytes.fromhex(salt), (binding["roster"]["sha256"] + ":" + job_id).encode(),
                             hashlib.sha256).hexdigest()
            hold = declaration_ok and release_hold(candidate)
            # Execution failures remain unavailable regardless of a later review.
            eligible = result["status"] == "complete" and declaration_ok and not hold
            packet = None
            if eligible:
                packet = {"schema": "card-fixed3000-blind-declaration-pair-v1", "blinded_id": blind,
                          "target_declaration": target, "candidate_declaration": candidate,
                          "public_context": contexts, "context_version": context_version,
                          "instruction": "Compare the declared statistical model, null, outcome, population and sample restrictions. Ignore estimator, inference procedure, code and presentation differences. Use supplied variable definitions for genuine shorthand, preserve explicit conflicts, and mark unresolved rather than infer from code, numerical behavior or abandoned alternatives."}
                new_packets.append(packet)
            observations[job_id] = {"job_id": job_id, "blinded_id": blind,
                "source_status": result["status"], "session_id": result.get("session_id"),
                "raw_match": raw_match if raw_valid else None, "raw_self_verdict": raw,
                "raw_label_status": "classified" if raw_valid else "unavailable",
                "declaration_status": "reviewable" if eligible else "public_release_hold" if hold else "unavailable",
                "result_sha256": result_sha, "raw_result": result,
                "closed_record_sha256": record["payload_sha256"],
                "source_pack": {"path": pack["path"], "index": pack["index"],
                                "export_receipt": pack["export_receipt"], "record": record},
                "public_pair_sha256": sha(packet) if packet else None}
            closed_seen[job_id] = record["payload_sha256"]
    out = Path(destination); need(not out.exists(), "Fresh immutable export directory required")
    out.mkdir(mode=0o700, parents=True)
    public_refs = []
    new_packets.sort(key=lambda p: p["blinded_id"])
    for offset in range(0, len(new_packets), 64):
        batch = new_packets[offset:offset+64]
        public_refs.append(write_once(out / "public" / ("batch-%03d.json" % (offset // 64 + 1)),
                                     {"schema": "card-fixed3000-blind-batch-v1", "pairs": batch,
                                      "pair_hashes": {r["blinded_id"]: sha(r) for r in batch}}))
    private_rows = []
    for position, row in enumerate(rows):
        state = observations.get(row["job_id"], {"job_id": row["job_id"],
            "source_status": "started_outcome_pending" if row["job_id"] in claimed else "not_observed",
            "raw_match": None, "raw_label_status": "unavailable", "declaration_status": "unavailable"})
        if not state.get("closed_record_sha256"):
            state = dict(state, source_status="started_outcome_pending" if row["job_id"] in claimed else "not_observed")
        merged = dict(row); merged.update(state)
        merged.update(position=position, shard=position % 4, target_id="original-001",
                      shared_world_cluster=row["world_id"])
        private_rows.append(merged)
    private = {"schema": "card-fixed3000-private-review-map-v1", "roster": binding["roster"],
               "original": binding["original"], "contexts": binding["contexts"],
               "context_version": context_version,
               "blinding_salt": salt, "rows": private_rows, "claimed_ids": sorted(claimed),
               "input_binding": binding_ref, "public_batches": public_refs,
               "previous_private": binding.get("previous_private"), "new_payload_bytes_read": read_bytes}
    private_ref = write_once(out / "private-map.json", private)
    return write_once(out / "export-receipt.json", {"schema": "card-fixed3000-review-export-receipt-v1",
        "private_map": private_ref, "public_batches": public_refs, "planned": 3000,
        "closed": len(closed_seen), "new_public_pairs": len(new_packets),
        "new_payload_bytes_read": read_bytes, "semantic_review": "pending", "scientific_computation": False})


def integrate(private_ref, review_refs, destination, final=False, terminal_ref=None):
    private = read(private_ref); need(private["schema"] == "card-fixed3000-private-review-map-v1", "Wrong private map")
    roster = roster_rows(read(private["roster"])); rows = private["rows"]
    need([r["job_id"] for r in rows] == [r["job_id"] for r in roster], "Exact one-to-one roster order required")
    terminal_by_id = {}
    if final:
        need(terminal_ref is not None, "Final integration requires executor terminal dispositions")
        terminal = read(terminal_ref)
        need(terminal["schema"] == "card-fixed3000-terminal-dispositions-v1" and
             terminal["roster"] == private["roster"] and terminal["all_workers_exited"] is True and
             terminal["no_reruns"] is True and len(terminal["rows"]) == 3000,
             "Terminal executor disposition receipt differs")
        terminal_by_id = {r["job_id"]: r for r in terminal["rows"]}
        need(len(terminal_by_id) == 3000 and set(terminal_by_id) == {r["job_id"] for r in rows},
             "Terminal inventory is not one-to-one")
        for row in rows:
            t = terminal_by_id[row["job_id"]]
            need(t["shard"] == row["shard"] and t["status"] in
                 {"complete", "unavailable", "unstarted_drain", "previously_started_no_retry",
                  "started_outcome_unavailable", "not_dispatched"}, "Unknown terminal status")
            if row.get("closed_record_sha256"):
                need(t["status"] == row["source_status"], "Closed and terminal status differ")
            else:
                need(t["status"] != "complete", "Terminal completion has no closed declaration evidence")
                if row["job_id"] in private["claimed_ids"]:
                    need(t["status"] not in {"unstarted_drain", "not_dispatched"},
                         "Started checkpoint cannot be relabeled unstarted")
    known = {r.get("blinded_id"): r for r in rows if r.get("blinded_id")}
    labels = {}; versions = {}; source_refs = []
    for ref in review_refs:
        review = read(ref); need(review["schema"] == "card-fixed3000-independent-labels-v1", "Wrong review schema")
        for label in review["labels"]:
            blind = label["blinded_id"]; row = known.get(blind)
            need(row and row["declaration_status"] == "reviewable" and
                 label["public_pair_sha256"] == row["public_pair_sha256"], "Review pair binding differs")
            need(label["label"] in {"match", "nonmatch", "unresolved"} and
                 all(isinstance(label.get(k), str) and label[k].strip() for k in
                     ("reason", "reviewer", "version", "context_version", "reviewed_at")),
                 "Explicit label and review provenance required")
            need(label["context_version"] == private["context_version"], "Review context version differs")
            prior = versions.setdefault(blind, [])
            need(not prior, "Each fresh pair receives exactly one independent review")
            if prior:
                need(label.get("supersedes_label_sha256") == sha(prior[-1]) and
                     label.get("adjudication_reason"), "Duplicate review needs explicit linked adjudication")
            else:
                need(not label.get("supersedes_label_sha256"), "Missing superseded label")
            need(label["version"] not in [x["version"] for x in prior], "Duplicate review version")
            prior.append(label); labels[blind] = label
        source_refs.append(ref)
    joined = []; raw_rows = []
    for row in rows:
        label = labels.get(row.get("blinded_id")); available = bool(label and label["label"] != "unresolved")
        valid = available and row["source_status"] == "complete" and row["declaration_status"] == "reviewable"
        if final and row["declaration_status"] == "reviewable":
            need(label is not None, "Final integration requires a disposition for every reviewable pair")
        base = {k: row[k] for k in ("job_id", "world_id", "stratum", "order_position", "position", "shard",
                                    "row_hash", "canonical_world_hash", "packet", "target_id", "shared_world_cluster")}
        joined.append(dict(base, reviewed_match=label["label"] == "match" if valid else None,
            reviewed_status="classified" if valid else "unresolved" if label else "review_pending" if row["declaration_status"] == "reviewable" else "unavailable",
            source_status=row["source_status"], declaration_status=row["declaration_status"],
            terminal_status=terminal_by_id.get(row["job_id"], {}).get("status"),
            result_sha256=row.get("result_sha256"), public_pair_sha256=row.get("public_pair_sha256"),
            label_source="independent_declaration_review" if label else None,
            review_provenance=versions.get(row.get("blinded_id"), [])))
        raw_rows.append(dict(base, raw_match=row["raw_match"], raw_status=row["raw_label_status"],
                             source_status=row["source_status"],
                             terminal_status=terminal_by_id.get(row["job_id"], {}).get("status"),
                             result_sha256=row.get("result_sha256")))
    out = Path(destination); need(not out.exists(), "Fresh versioned integration directory required")
    out.mkdir(mode=0o700, parents=True)
    review_ref = write_once(out / "reviewed-labels.json", {"schema": "card-fixed3000-reviewed-labels-v1",
        "roster": private["roster"], "original": private["original"], "final_dispositions": final, "rows": joined})
    raw_ref = write_once(out / "raw-labels.json", {"schema": "card-fixed3000-raw-labels-v1",
        "roster": private["roster"], "original": private["original"], "rows": raw_rows})
    return write_once(out / "integration-receipt.json", {"schema": "card-fixed3000-review-integration-v1",
        "private_map": private_ref, "review_inputs": source_refs, "reviewed_labels": review_ref,
        "raw_labels": raw_ref, "planned": 3000, "final_dispositions": final,
        "terminal_dispositions": terminal_ref,
        "raw_labels_preserved_separately": True, "manager_owned_semantic_review": True,
        "versioned": True, "scientific_computation": False})


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="action", required=True)
    e = sub.add_parser("export"); e.add_argument("--binding", required=True); e.add_argument("--binding-sha256", required=True)
    e.add_argument("--output", required=True)
    i = sub.add_parser("integrate"); i.add_argument("--private-map", required=True); i.add_argument("--private-map-sha256", required=True)
    i.add_argument("--reviews-manifest", required=True); i.add_argument("--reviews-manifest-sha256", required=True)
    i.add_argument("--output", required=True); i.add_argument("--final", action="store_true")
    i.add_argument("--terminal-dispositions"); i.add_argument("--terminal-dispositions-sha256")
    a = p.parse_args()
    if a.action == "export":
        receipt = export({"path": a.binding, "sha256": a.binding_sha256}, a.output)
    else:
        refs = read({"path": a.reviews_manifest, "sha256": a.reviews_manifest_sha256})
        terminal = {"path": a.terminal_dispositions, "sha256": a.terminal_dispositions_sha256} if a.terminal_dispositions else None
        receipt = integrate({"path": a.private_map, "sha256": a.private_map_sha256}, refs["reviews"], a.output, a.final, terminal)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
