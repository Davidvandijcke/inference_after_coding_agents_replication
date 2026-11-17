#!/usr/bin/env python3
"""Checks every prompt display of prompts.tex against the prompt files in the package; stops on a mismatch.

Each passage of a display, between its omission marks [...], must occur in order in the file the cohort received
(SOURCES); the displays in WHOLE must equal their file. Pages 4 and 6 to 8 show an earlier cohort's prompts, which
are not shipped, and must equal pages 14 to 17 with AD061 for AD055. Page 10 has no file and is pinned by its
SHA-256. Page 22 must equal the prompt that the Card review script assembles (CARD_REVIEW), with the display's
two placeholders; its two definitions are read from the script's syntax tree and the script is not run.
"""
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = "data/agent_runs/"
MENU = RUNS + "issp_menu_anti/inputs/"
PLACEBO = "code/03_placebo/data_generation/runtime/"
# every shipped copy of the commitment and comparison prompts
SHARED = ["code/02_open_ended/data_generation/prompts/", PLACEBO, RUNS + "card_luna/packet/",
          RUNS + "card_sonnet_masked/packet/", MENU, RUNS + "issp_request_luna/inputs/"]
SOURCES = {  # page: the files that hold the displayed prompt
    1: ["code/02_open_ended/data_generation/prompts/research_prompt.txt",
        "code/10_self_correction/data_generation/four_agent_null_run/frozen/research_prompt.txt"],
    2: ["data/raw/pathways/CODEBOOK.md"],
    3: [MENU + "developer-instructions.txt"],
    5: [MENU + "prompts/frontier-anti-analysis.txt"],
    9: [RUNS + "card_luna/packet/research-prompt.txt"],
    11: [folder + "commitment_prompt.txt" for folder in SHARED],
    12: [folder + "comparison_prompt.txt" for folder in SHARED],
    13: [RUNS + "issp_request_luna/inputs/research_prompt.txt"],
    14: [MENU + "prompts/fixed_reference-anti-analysis.txt"],
    15: [MENU + "prompts/client_pressure-anti-analysis.txt"],
    16: [MENU + "prompts/client_pressure-anti-aligned.txt"],
    17: [MENU + "prompts/client_pressure-anti-pressure.txt"],
    18: [PLACEBO + "research_prompt.txt"],
    19: [PLACEBO + "codebooks/wellness.md"],
    20: [PLACEBO + "codebooks/teachers.md"],
    21: [PLACEBO + "codebooks/consumers.md"],
}
# page 22: the script that assembled the reviewer's prompt, what it was given, and the packet builder
CARD_REVIEW = "code/09_card_agents/data_generation/terra/rereview_gl.py"
CARD_REVIEW_GIVEN = RUNS + "card_luna/review_prompt.json"
CARD_REVIEW_PACKETS = "code/09_card_agents/data_generation/terra/rereview_packets.py"
ASSEMBLED = {22}
WHOLE = {1, 11}                                  # displays that quote their file from start to end
EARLIER_COHORT = {4: 14, 6: 15, 7: 16, 8: 17}    # page: the shipped cohort's page it must equal with AD061
PINNED = {10: "ac6e2667182dd5fee1edd33c06fd8de4ac2d1d0e79273776b8fd2d61866f2ea5"}
PAGES = 22
OMISSION = "[\\ldots]"


def plain(text):
    """Text of a display or a prompt file without markup, white space normalized."""
    text = re.sub(r"\\texttt\{((?:[^{}\\]|\\.)*)\}", r"\1", text)
    text = text.replace("\\_", "_").replace("$q$", "q").replace("$t$", "t").replace("`", "")
    return re.sub(r"\s+", " ", text).strip()


def missing_passages(display, source):
    """Passages of the display, between its omission marks, that the file does not hold in that order."""
    # the comparison prompt's file holds a placeholder where the display names what is inserted
    source = plain(source).replace("{{TARGET_JSON}}", "[original commitment]")
    position, missing = 0, []
    for passage in (plain(p) for p in display.split(OMISSION)):
        found = source.find(passage, position)
        if passage and found < 0:
            missing.append(passage)
        elif passage:
            position = found + len(passage)
    return missing


def card_review_prompt():
    """The Card match reviewer's prompt as its script assembles it, with the display's placeholders for the inserts."""
    tree = ast.parse((ROOT / CARD_REVIEW).read_text(encoding="utf-8"))
    keep = [node for node in tree.body
            if (isinstance(node, ast.Assign) and [t.id for t in node.targets if isinstance(t, ast.Name)] == ["RULE"])
            or (isinstance(node, ast.FunctionDef) and node.name == "prompt_for")]
    if len(keep) != 2:
        sys.exit(f"{CARD_REVIEW}: RULE and prompt_for not found once each")
    scope = {"json": json}
    exec(compile(ast.Module(body=keep, type_ignores=[]), CARD_REVIEW, "exec"), scope)
    given = json.loads((ROOT / CARD_REVIEW_GIVEN).read_text(encoding="utf-8"))
    prompt = scope["prompt_for"]({"instruction": given["instruction"], "data_facts": given["data_facts"],
                                  "target_declaration": None, "pairs": [], "public_context": []})
    head, found, inserts = prompt.partition("Public context:\n\n")
    if not found or not inserts.startswith("\n\nPacket:\n\n"):
        sys.exit(f"{CARD_REVIEW}: the prompt no longer ends with the public context and the packet")
    size = re.search(r"^SEED, BATCH = \d+, (\d+)$", (ROOT / CARD_REVIEW_PACKETS).read_text(encoding="utf-8"), re.M)
    if not size:
        sys.exit(f"{CARD_REVIEW_PACKETS}: packet size not found")
    return (head + "Public context: [study description and codebook]\n\nPacket: [the original analysis's declaration "
            f"and {size.group(1)} blinded replay declarations]")


def main():
    tex = (ROOT / "code" / "11_prompts" / "prompts.tex").read_text(encoding="utf-8")
    displays = dict(enumerate(re.findall(r"\\begin\{promptbox\}\n(.*?)\n\\end\{promptbox\}", tex, re.S), 1))
    faults = []
    if len(displays) != PAGES:
        faults.append(f"prompts.tex has {len(displays)} displays, not {PAGES}; paper/main.tex includes them by page")
    if set(SOURCES) | set(EARLIER_COHORT) | set(PINNED) | ASSEMBLED != set(range(1, PAGES + 1)):
        faults.append("a page has no rule in SOURCES, EARLIER_COHORT, PINNED or ASSEMBLED")
    for page, display in displays.items():
        if "\\" in plain(display.replace(OMISSION, " ")):
            faults.append(f"page {page}: markup the check does not remove: {plain(display)[:80]}")
        for path in SOURCES.get(page, []):
            source = (ROOT / path).read_text(encoding="utf-8")
            if page in WHOLE and plain(display) != plain(source):
                faults.append(f"page {page}: the display is not the whole of {path}")
            for passage in missing_passages(display, source):
                faults.append(f"page {page}: not in {path}: {passage[:160]}")
        if page in EARLIER_COHORT:
            twin = EARLIER_COHORT[page]
            if plain(display).replace("AD055", "AD061") != plain(displays.get(twin, "")):
                faults.append(f"page {page}: differs from page {twin} by more than the model's name")
        if page in ASSEMBLED:  # this display writes quotation marks in LaTeX; the prompt has straight ones
            if plain(display.replace("``", '"').replace("''", '"')) != plain(card_review_prompt()):
                faults.append(f"page {page}: the display is not the prompt that {CARD_REVIEW} assembles")
        if page in PINNED and hashlib.sha256(plain(display).encode("utf-8")).hexdigest() != PINNED[page]:
            faults.append(f"page {page}: the display's text has changed (no file in the package to check it against)")
    if faults:
        sys.exit("prompt displays do not match the prompt files:\n  " + "\n  ".join(faults))
    print(f"{len(displays)} prompt displays checked: {len(SOURCES)} against {sum(map(len, SOURCES.values()))} prompt files, "
          f"{len(EARLIER_COHORT)} against their AD061 pages, {len(PINNED)} against its pinned text, "
          f"{len(ASSEMBLED)} against the prompt its review script assembles")


if __name__ == "__main__":
    main()
