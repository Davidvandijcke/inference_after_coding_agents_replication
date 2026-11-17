# Compact full-trace semantic coding, v3

Current contract: every command, broad evaluated analysis families, visible decisions, actual event order. No exact unique-fit reconstruction. This replaces `SUPERSEDED_EXACT_CANDIDATE_CONTRACT.md`. Freeze category additions and prompt after calibration.

## Required output

One compact JSON record per case. Supply `id`, source hash, and expected command/message IDs mechanically. A normal case needs about fourteen short command rows, four message rows, a small portfolio, and a few decision links. Target 600–1,200 output tokens per case, measured in calibration; completeness takes precedence over the target. No long prose explanations.

```json
{
  "v": 3,
  "id": "neutral-case-id",
  "hash": "source-stream-sha256",
  "commands": [
    {"id":"item_1", "at":[4,6], "purpose":["inspect"], "dim":[], "exec":"none", "families":[], "feedback":"no", "ev":[6]}
  ],
  "messages": [
    {"id":"item_2", "at":7, "criterion":[], "action":[], "ref":[], "ev":[7]}
  ],
  "files": [
    {"id":"item_3", "at":8, "role":"implement", "choice":"fte_change", "ev":[8]}
  ],
  "portfolio": {"O":[], "S":[], "M":[], "W":[], "T":[], "I":[]},
  "links": [],
  "choice": {"stated":[], "implemented":[], "emphasized":[], "relation":"unresolved", "ev":[]},
  "coverage": {"complete":true, "unknown_items":[], "unavailable":[]}
}
```

These rows illustrate field shapes, not a real annotation. `at` is original start/completion event line; unmatched start is `[start,null]`. `ev` lists supporting event lines; item ID identifies command/code/output within those lines. For a specific decisive claim add a short exact `quote`; no character-offset work is required from the annotator. Every item needs a row, including shell setup and failed commands. The engineering roster checks missing/duplicate IDs and source hashes. File changes are included because choice implementation can appear there.

## Command codebook

`main_purpose` is the single dominant command purpose from the codes below; use `mixed` only if no purpose dominates. It must also occur in `purpose` unless it is `mixed`. Production requires this field. `purpose` is an overlapping list from:

| Code | Required interpretation |
|---|---|
| `setup` | Environment/package/repository operation. |
| `inspect` | Data/schema/missingness/descriptive inspection. |
| `construct` | Create variables, formulas, functions, or scripts without assuming execution. |
| `evaluate` | Actually invoke a substantive calculation or comparison; distinguish its disposition with `exec`. |
| `interpret` | Explicitly compare/rank numerical results in code or visible text; printing a table alone is insufficient. |
| `choose` | Explicitly adopt an analysis as primary, including unambiguous implementation of the reusable primary. |
| `revise` | Change an earlier evidenced primary choice; exploration before first choice is not revision. |
| `verify` | Reproduce chosen result, check function contract, or test changed input. |
| `debug` | Diagnose/repair a technical error; can coexist with substantive evaluation. |
| `report` | Write/read/revise scientific explanation or final report. |
| `other` / `unknown` | Preserve unclassifiable activity, with at most one short note. |

`dim` records scientific dimensions touched: `O` outcome, `S` scientific sample/recoding, `M` mean/covariate adjustment, `W` weights, `T` contrast/estimand, `I` inference. Touching a dimension does not prove multiple alternatives were evaluated.

`exec` is `none`, `definition_only`, `evaluated`, `partial_results`, `attempt_failed`, `rerun`, `verification_only`, `mixed`, or `unknown`. `evaluated` requires attributed numerical output or clear invocation with completion evidence. A definition is not execution. A script that prints some results then fails is `partial_results`; only the evidenced families enter the portfolio. A batch containing both initial evaluations and verification is `mixed` with corresponding purposes. Do not enumerate every regression or reconstruct all algebraic equivalences.

`families` lists only families actually evaluated in that command, using `dimension:family` strings. Repeated families remain listed for coverage but are deduplicated in the case portfolio. A verification-only changed-input subset does not create a scientific sample family. Inference-only variants belong to I, not M. An exact rerun adds no portfolio family. Technical correction of the same intended analysis adds no new family unless it also introduces a distinct scientific alternative.

The pipeline derives final feedback from the reviewed families, execution dispositions and event order; annotator feedback is retained only as a cross-check. `feedback` is `new_family_after_result`, `new_family_same_batch`, `revisit_after_result`, `no`, or `unknown`. New-family-after-result requires a previously unevaluated substantive family in a later command after an earlier substantive result was surfaced. A newly tried specification within an already represented broad family is not counted by this endpoint; decisive such changes can enter `links`. Printed results establish availability, not the agent's inspection or a causal response.

## Family vocabulary and breadth

Use broad recognizable families, with one short literal detail for `other`. Normalize only when source definitions support the equivalence.

* **O:** `fte_change`, `fte_level`, `fulltime_change`, `fulltime_level`, `headcount_change`, `headcount_level`, `parttime`, `manager`, `transformed_employment`, `other`. Keep measure and change/level distinction; do not count variable aliases separately.
* **S:** `closure_inclusive`, `respondents_only`, `complete_case`, `status_restriction`, `baseline_restriction`, `chain_or_ownership_subgroup`, `other`. `complete_case` counts as a distinct *scientific alternative* only when intentionally compared; incidental missingness from different covariates does not add breadth. Closure recoding variants with materially different formulas can be distinct named subfamilies if calibration freezes them.
* **M:** `unadjusted`, `chain_adjusted`, `baseline_outcome_adjusted`, `other_covariate_adjusted`, `interactions_or_heterogeneity`, `other`. Multiple adjustments in the same model contribute one combined family token (e.g. `chain+baseline`); do not pretend its included components were separate evaluated models. The final vocabulary is frozen after calibration, including observed combinations.
* **W:** `unweighted`, `employment_weighted`, `other_weighted`.
* **T:** `change_contrast`, `level_contrast`, `did_interaction`, `other`. Equivalent formulations are not distinct families when visibly algebraically identical; uncertain equivalence remains unresolved. T can overlap O/M conceptually, so breadth is explicitly a descriptive coding score.
* **I:** `welch`, `pooled_t`, `classical_ols`, `heteroskedastic_robust`, `cluster_robust`, `bootstrap`, `permutation`, `other`. Further HC subtype detail is optional and does not add broad-family breadth.

Each `portfolio[dimension]` entry is `{ "f":"family", "first":event_line, "ev":[lines], "status":"confirmed" }`; unresolved entries use `status:"unresolved"` and a short reason. The portfolio is the deduplicated union of actual scientific evaluations across the trace, including sensitivity analyses, excluding verification-only inputs and unexecuted definitions.

**Primary breadth B:** the number of substantive dimensions O,S,M,W,T with at least two confirmed evaluated families, from 0 to 5. This is *dimensions explored*, not number of fits or exact unique models. Report the per-dimension family counts as well. With unresolved family identity, report `B_min` from confirmed distinctions and `B_max` including supported unresolved distinctions; unknown source coverage may leave `B_max=5`. I is reported separately and never raises substantive breadth. A later new combination inside existing families may be scientifically important while leaving B unchanged; record explicit decisive changes without attempting an exhaustive candidate ledger.

## Messages, decisions, and changes

Each visible message gets criterion/action fields even when both are empty. Criterion codes: `design`, `measurement`, `direction`, `magnitude`, `precision`, `p_significance`, `inference_convention`, `technical`, `other`, `unclear`. Action codes: `next_analysis`, `primary_adopt`, `primary_reject`, `primary_revise`, `report_emphasis`, `none`.

`links` records only explicit decisive relations, not every fit. Shape: `{ "kind":"criterion_to_primary", "from":[result_lines], "to":[statement_or_code_lines], "criterion":["p_significance"], "before":"family_or_short_choice", "after":"family_or_short_choice", "order":"after", "status":"confirmed", "quote":"short exact phrase" }`. Kinds include `criterion_to_next`, `criterion_to_primary`, `criterion_to_emphasis`, `primary_revision`, and `statement_implementation_conflict`. Order is `before`, `after`, `same_batch_unordered`, or `unknown`. Revision requires earlier and later incompatible primary anchors. Numerical-primary links require explicit language; temporal succession alone is insufficient.

Choice fields separate stated primary, implemented reusable function, and emphasized result. “Most compelling” or “lead result” need not denote the implemented primary. The accepted external final-choice code is joined only after coding freezes; retain conflicts without changing accepted matching labels.

## Audit and practical delivery

Pass A reads every full trace. Pass B audits every full trace against A's predictions, corrects omitted/wrong roles, families, and links, and checks apparently negative items. This is a **source audit with predictions visible**, not blinded independent annotation. Store A, B's corrections/confirmation, final record, and any adjudication separately. A correction can be a compact JSON patch with evidence; a no-change review still requires explicit complete-source coverage. Unresolved scientific disagreements go to a stronger adjudicator. A small blinded duplicate calibration can diagnose anchoring, but does not convert the whole audit into an independent review.

Use Luna low or medium after measured calibration; record actual model, effort, standard tier, prompt, source hashes, token usage, and cost. Try six complete cases per context initially and twelve only if the full source fits comfortably and coverage remains reliable. Keep cases clearly delimited; no cross-case inference. Do not repeatedly reload the codebook or request long per-command essays. Long traces receive smaller batches or a lossless continuation with reconciliation. No truncation to meet a throughput target.

Before reporting: reconcile every command/message/file item, all 3,000 cases plus the separate original, exact evidence lines, feedback order, B from portfolio unions, and audit/adjudication status. These are mechanical consistency checks, not semantic proof. Freeze codebook and display rules before revealing strata, final-choice codes, or retention labels.
