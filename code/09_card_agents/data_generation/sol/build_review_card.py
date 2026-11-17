"""Build the blind read of the Sol Card replay reports under the Card facts rule. No model calls.

Reads every case archive of the Sol Card cohort, takes the last agent message of the analysis phase,
and writes review7/cases.jsonl with opaque ids in one random order; the key (id to case_id) goes to
review7/private/. No read sees a case id, stratum or verdict. 40 reads are repeated.

Usage: sbatch analysis.sbatch build_review_card.py
"""
import glob
import hashlib
import io
import json
import random
import secrets
import tarfile

import common
from card_correction_scan import last_message

R = common.RUN
SRC = "<cluster-project>/sol-card-main-gl/out/*/cases-*.tar"
RETEST = 40


def reports():
    for f in sorted(glob.glob(SRC)):
        with tarfile.open(f) as outer:
            for m in outer.getmembers():
                if m.name.endswith(".tar.gz"):
                    with tarfile.open(fileobj=io.BytesIO(outer.extractfile(m).read()), mode="r:gz") as t:
                        txt = last_message(t)
                    if txt:
                        yield m.name.split("-")[0], txt


def main():
    rule = (common.HERE / "review_rule_facts_card.txt").read_text()
    salt = secrets.token_hex(16)
    cases, key = [], {}
    for case_id, txt in reports():
        cid = hashlib.sha256(f"{salt}:{case_id}".encode()).hexdigest()[:20]
        cases.append({"id": cid, "task": "facts_card", "read": 1, "prompt": rule.replace("{{TEXT}}", txt.strip())})
        key[cid] = {"case_id": case_id}
    rng = random.Random(20261036)
    rng.shuffle(cases)
    cases += [dict(c, read=2) for c in rng.sample(cases, RETEST)]
    rng.shuffle(cases)
    out = R / "review7"
    (out / "private").mkdir(parents=True, exist_ok=True)
    (out / "private").chmod(0o700)
    with open(out / "cases.jsonl", "w") as f:
        for i, c in enumerate(cases):
            f.write(json.dumps(dict(c, position=i)) + "\n")
    common.write_json(out / "private" / "key.json", key)
    print(json.dumps({"cases": len(cases), "reports": len(key)}))


if __name__ == "__main__":
    main()
