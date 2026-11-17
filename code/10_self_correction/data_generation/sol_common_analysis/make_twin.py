"""No-explore wedge runs: frozen datasets and sealed rosters. No model calls.

Plan: docs/WEDGE_GOAL_2026-11-01.md and the design document's frozen plan. Prompts and codebook are the no-explore arm's, byte for byte. The moderate effect is the power panel's latent shift of 0.28 on employment at 12 months; at 0 the generator reproduces the frozen one.
Usage: python3 make_twin.py COHORT MODEL OUT MULTI_MODEL_CODE POWER_SOURCE [DATA_ROOT]   (COHORT is stage0, panel, panel2, nulls or vignette, whose sessions all use one dataset; DATA_ROOT reuses another model's datasets)
"""
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

COHORTS = {"stage0": (9_980_000, 12, 0.28, 20261061), "panel": (9_990_000, 30, 0.28, 20261062), "nulls": (8_100_000, 200, 0.0, 20261063), "panel2": (9_990_030, 30, 0.28, 20261065), "vignette": (8_400_000, 200, 0.0, 20261066)}
PREFIX = {"research_prompt.txt": "59306d5788438eb2", "commitment_prompt.txt": "f2a0efaf0017105b", "CODEBOOK.md": "36753982e29ed177"}


def main():
    cohort, model = sys.argv[1:3]
    out, code, source = Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5])
    data = Path(sys.argv[6]) if len(sys.argv) > 6 else None
    seed, n, tau, order_seed = COHORTS[cohort]
    sys.path.insert(0, str(source))
    import generate
    for d in ("frozen", "logs", "private"):
        (out / d).mkdir(parents=True, exist_ok=True)
    if not (out / "code").is_symlink():
        (out / "code").symlink_to(code)
    for name, prefix in PREFIX.items():
        shutil.copy2(code / "frozen_noexplore" / name, out / "frozen" / name)
        assert hashlib.sha256((out / "frozen" / name).read_bytes()).hexdigest().startswith(prefix), name
    shutil.copy2(source / "generate.py", out / "frozen" / "generate.py")
    if data:
        if not (out / "data").is_symlink():
            (out / "data").symlink_to(data / "data")
    else:
        (out / "data/originals").mkdir(parents=True, exist_ok=True)
    order = list(range(n))
    random.Random(order_seed).shuffle(order)
    roster = []
    for position, slot in enumerate(order):
        one = cohort == "vignette"
        name = f"{cohort}.csv" if one else f"{cohort}-{slot:03d}.csv"
        if not data and not (one and (out / "data/originals" / name).exists()):
            generate.write_csv(generate.generate(seed if one else seed + slot, 2000, tau), out / "data/originals" / name)
        roster.append({"case_id": hashlib.sha256(f"twin-{cohort}:{model}:{slot}".encode()).hexdigest()[:32], "model": model, "slot": slot, "seed": seed if cohort == "vignette" else seed + slot, "data": name, "position": position})
    (out / "roster.json").write_text(json.dumps({"cases": roster, "order_seed": order_seed}, indent=1))
    print(cohort, model, len(roster))


if __name__ == "__main__":
    main()
