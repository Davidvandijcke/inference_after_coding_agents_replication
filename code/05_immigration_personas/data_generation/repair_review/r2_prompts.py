#!/usr/bin/env python3
"""Prompts for the second step of Miao et al.'s pipeline: a structured summary of each repaired analysis, then a
PASS/FAIL review of that summary. No model calls.

Verbatim from the paper (Appendix B, extracted from lit/.../2026_Miao_AgenticGardenForkingPaths.pdf):
  prompts/review_appendixB.txt          the review prompt given to the AI and human reviewers
  prompts/summary_examples_appendixB.txt the two example specification summaries
Reconstructed, because the authors' review harness is not public (their repository at 7c8925f, still the latest
commit on 4 November 2026, has only the repair step):
  SUMMARY_INSTRUCTIONS  who writes the summary and how; the paper says only that reviewers saw "the same structured
                        summary" covering outcome, exposure, sample restrictions, controls, model and inference
  FORMAT_INSTRUCTION    the paper says the AI reviewer "additionally received a fixed output-format instruction (a
                        binary PASS/FAIL verdict with any flaws listed)" but does not print it
One factual edit for the corrected file: its respondent count (154,921) replaces the paper's 152,594.
"""
from pathlib import Path

P = Path(__file__).resolve().parent / "prompts"
REVIEW = (P / "review_appendixB.txt").read_text().strip()
EXAMPLES = (P / "summary_examples_appendixB.txt").read_text().strip()
RESPONDENTS = {"miao": "152,594", "clean": "154,921"}

SUMMARY_INSTRUCTIONS = """You will write a structured summary of one statistical analysis, to be read by a methodological reviewer who sees nothing else about it.

Files:
- Analysis script: {script}
- Its output: {results}
- Data: {data}

Read the script, and run it or inspect the data as needed to state the sample exactly (survey waves, the countries used with their number, and the number of observations or cells). Then write the summary in exactly the format of the two examples below, with the sections Regression, Estimator and standard errors, Outcome, Exposure, Sample and Covariates. Describe what the script does, plainly and factually, naming variables as the examples do. Do not evaluate the analysis, do not report its estimate, standard error, confidence interval or p-value, and do not mention how or by whom the script was written.

Write the summary, and nothing else, to {out}.

{examples}"""

FORMAT_INSTRUCTION = """Answer on the first line with exactly `VERDICT: PASS` or `VERDICT: FAIL`. If you mark FAIL, list the specific problem(s) below the verdict. If you mark PASS you may optionally note minor concerns below it."""


def summary_prompt(script, results, data, out):
    return SUMMARY_INSTRUCTIONS.format(script=script, results=results, data=data, out=out, examples=EXAMPLES)


def review_prompt(summary, data_file):
    review = REVIEW.replace("152,594", RESPONDENTS[data_file])
    return f"{review}\n\n{FORMAT_INSTRUCTION}\n\n**Specification**\n\n{summary.strip()}\n"


def verdict(text):
    first = next((l.strip().strip("`*") for l in text.splitlines() if l.strip()), "")
    return {"VERDICT: PASS": "PASS", "VERDICT: FAIL": "FAIL"}.get(first.upper(), None)


if __name__ == "__main__":  # print both prompts for inspection
    print(summary_prompt("<script>", "<results>", "<data>", "<out>"))
    print("\n" + "=" * 80 + "\n")
    print(review_prompt("<summary>", "miao"))
