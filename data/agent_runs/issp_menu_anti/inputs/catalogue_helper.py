#!/usr/bin/env python3
"""Fit one or all configurations in the public 108-model catalogue."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import warnings
from pathlib import Path
from typing import Any

ALL_STOCKS = ["migstock_wb", "migstock_un", "migstock_oecd"]
EXPECTED_CATALOGUE_SHA256 = "c6b5a2aed74edc352aaacd31d6ccf06588fd780b4700e798562493eed5688a5c"


class StructuralNonestimability(Exception):
    """A prespecified design cannot produce a valid HC3 t statistic."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def unavailable(model: dict[str, Any], classification: str, reason: str, caught: list[str],
                diagnostics: dict[str, Any] | None = None, reason_code: str | None = None) -> dict[str, Any]:
    structural = classification == "structural_nonestimability"
    result = {
        "model_id": model["model_id"], "sequence": model["sequence"],
        "outcome": model["outcome"], "exposure": model["exposure"],
        "coverage": model["coverage"], "control_set": model["control_set"],
        "working_model": model["working_model"], "null_hypothesis": model["null_hypothesis"],
        "population": model["population"], "sample_restrictions": model["sample_restrictions"],
        "raw_status": "unavailable", "availability_classification": classification,
        "unavailable_reason_code": reason_code, "unavailable_reason": reason,
        "n_countries": None, "country_ids": [], "years_by_country": {},
        "design_rank": None, "residual_df": None, "exposure_variance": None,
        "max_leverage": None, "condition_number": None,
        "slope": None, "hc3_se": None, "hc3_t": None, "raw_hc3_p": None,
        "classical_se": None, "classical_t": None, "classical_p": None,
        "test_inference_p": 1.0 if structural else None, "catalogue_m": None, "warnings": caught,
    }
    if diagnostics:
        result.update(diagnostics)
    return result


def fit_model(data: Any, model: dict[str, Any], np: Any, pd: Any, stats: Any, sm: Any) -> dict[str, Any]:
    caught_text: list[str] = []
    result = unavailable(model, "unexpected_error", "not attempted", caught_text)
    diagnostics: dict[str, Any] = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            value_cols = [model["outcome"], model["exposure"], *model["controls"]]
            required = ["iso_country", "year", *value_cols]
            if model["coverage"] == "all_three_observed":
                required.extend(ALL_STOCKS)
            required = list(dict.fromkeys(required))
            missing = [column for column in required if column not in data.columns]
            if missing:
                raise KeyError(f"required columns absent: {missing}")
            complete = data.loc[:, required].dropna(axis=0, how="any")
            cy = complete.groupby(["iso_country", "year"], as_index=False, sort=True)[value_cols].mean()
            country = cy.groupby("iso_country", as_index=False, sort=True)[value_cols].mean()
            n = int(len(country))
            country_ids = country["iso_country"].astype(str).tolist()
            years_by_country = {
                str(country_id): sorted(group["year"].drop_duplicates().tolist())
                for country_id, group in cy.groupby("iso_country", sort=True)
            }
            variance = float(np.var(country[model["exposure"]].to_numpy(dtype=float), ddof=0)) if n else math.nan
            diagnostics = {"n_countries": n, "country_ids": country_ids, "years_by_country": years_by_country,
                           "design_rank": None, "residual_df": None,
                           "exposure_variance": variance if math.isfinite(variance) else None}
            if n == 0:
                raise StructuralNonestimability("no_complete_countries")
            if not math.isfinite(variance) or variance <= 0:
                raise StructuralNonestimability(f"exposure_variance_invalid: {variance}")
            y = country[model["outcome"]].to_numpy(dtype=float)
            X = sm.add_constant(country[[model["exposure"], *model["controls"]]].to_numpy(dtype=float), has_constant="add")
            rank = int(np.linalg.matrix_rank(X))
            df = int(n - rank)
            diagnostics.update(design_rank=rank, residual_df=df)
            if rank < X.shape[1]:
                raise StructuralNonestimability(f"rank_deficient: rank={rank}, columns={X.shape[1]}")
            if df <= 0:
                raise StructuralNonestimability(f"nonpositive_residual_df: {df}")
            fitted = sm.OLS(y, X).fit()
            leverage = fitted.get_influence().hat_matrix_diag
            max_leverage = float(np.max(leverage))
            condition_number = float(np.linalg.cond(X))
            diagnostics.update(max_leverage=max_leverage,
                               condition_number=condition_number if math.isfinite(condition_number) else None)
            if not math.isfinite(max_leverage) or max_leverage >= 1 - 1e-12:
                raise StructuralNonestimability(f"hc3_unit_leverage: max_leverage={max_leverage}")
            robust = fitted.get_robustcov_results(cov_type="HC3", use_t=False)
            slope = float(fitted.params[1])
            covariance = np.asarray(robust.cov_params(), dtype=float)
            target_variance = float(covariance[1, 1])
            if not math.isfinite(slope) or not np.all(np.isfinite(covariance)):
                raise StructuralNonestimability("nonfinite_hc3: covariance or slope is nonfinite")
            if not math.isfinite(target_variance) or target_variance <= 0:
                raise StructuralNonestimability(f"nonpositive_target_variance: {target_variance}")
            se = math.sqrt(target_variance)
            t_value = slope / se
            p_value = float(2 * stats.t.sf(abs(t_value), df))
            if not all(math.isfinite(v) for v in [se, t_value, p_value]) or se <= 0:
                raise StructuralNonestimability(f"nonfinite_hc3: se={se}, t={t_value}, p={p_value}")
            classical_se_raw = float(fitted.bse[1])
            classical_se = classical_se_raw if math.isfinite(classical_se_raw) and classical_se_raw > 0 else None
            classical_t = slope / classical_se if classical_se is not None else None
            classical_p = float(2 * stats.t.sf(abs(classical_t), df)) if classical_t is not None else None
            if classical_p is not None and not math.isfinite(classical_p):
                classical_p = None
            if classical_se is None or classical_p is None:
                caught_text.append("secondary_classical_inference_unavailable")
            result = {
                "model_id": model["model_id"], "sequence": model["sequence"],
                "outcome": model["outcome"], "exposure": model["exposure"],
                "coverage": model["coverage"], "control_set": model["control_set"],
                "working_model": model["working_model"], "null_hypothesis": model["null_hypothesis"],
                "population": model["population"], "sample_restrictions": model["sample_restrictions"],
                "raw_status": "available", "availability_classification": "estimable",
                "unavailable_reason_code": None, "unavailable_reason": None, **diagnostics,
                "slope": slope, "hc3_se": se, "hc3_t": t_value, "raw_hc3_p": p_value,
                "classical_se": classical_se, "classical_t": classical_t, "classical_p": classical_p,
                "test_inference_p": p_value, "catalogue_m": None, "warnings": [],
            }
        except StructuralNonestimability as exc:
            message = str(exc)
            result = unavailable(model, "structural_nonestimability", message, caught_text,
                                 diagnostics=diagnostics, reason_code=message.split(":", 1)[0])
        except Exception as exc:
            result = unavailable(model, "unexpected_error", f"{type(exc).__name__}: {exc}", caught_text,
                                 diagnostics=diagnostics, reason_code="unexpected_error")
    caught_text.extend(f"{item.category.__name__}: {item.message}" for item in caught)
    result["warnings"] = caught_text
    return result


def add_catalogue_m(fits: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [fit for fit in fits if fit["raw_status"] == "available" and
                math.isfinite(fit["hc3_t"])]
    if eligible:
        z_values = [fit["hc3_t"] for fit in eligible]
        denominator = len(z_values)
        for fit in eligible:
            z_value = fit["hc3_t"]
            lower = sum(value <= z_value for value in z_values) / denominator
            upper = sum(value >= z_value for value in z_values) / denominator
            fit["catalogue_m"] = min(1.0, 2 * min(lower, upper))
    return {"planned_paths": len(fits), "eligible_finite_z": len(eligible),
            "unavailable_paths": len(fits) - len(eligible),
            "pool_status": "available" if eligible else "unavailable"}


def load_scientific_stack() -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    import numpy as np
    import pandas as pd
    import scipy
    from scipy import stats
    import statsmodels
    import statsmodels.api as sm
    receipt = {
        "python": platform.python_version(), "python_executable": sys.executable,
        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "module_paths": {"numpy": np.__file__, "pandas": pd.__file__, "scipy": scipy.__file__,
                         "statsmodels": statsmodels.__file__},
    }
    return np, pd, stats, sm, receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalogue", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true")
    group.add_argument("--model-id")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite output")
    if sha256(args.catalogue) != EXPECTED_CATALOGUE_SHA256:
        raise RuntimeError("catalogue SHA-256 mismatch")
    catalogue = json.loads(args.catalogue.read_text(encoding="utf-8"))
    models = catalogue["models"]
    selected = models if args.all else [m for m in models if m["model_id"] == args.model_id]
    if not selected:
        raise RuntimeError(f"unknown model ID: {args.model_id}")
    np, pd, stats, sm, runtime = load_scientific_stack()
    data = pd.read_csv(args.data)
    fits = [fit_model(data, model, np, pd, stats, sm) for model in selected]
    m_pool = add_catalogue_m(fits) if args.all else {
        "planned_paths": 108, "eligible_finite_z": None, "unavailable_paths": None,
        "pool_status": "unavailable_single_fit_does_not_define_common_catalogue_pool",
    }
    payload = {
        "schema": "public-catalogue-fit-output-v1", "catalogue_sha256": sha256(args.catalogue),
        "data_sha256": sha256(args.data), "runtime": runtime,
        "fit_scope": "all" if args.all else args.model_id, "m_pool": m_pool, "fits": fits,
        "policy_note": "Enumerated structural domain failures retain unavailable raw theta/SE/p and receive separate test_inference_p=1. Unexpected errors retain test_inference_p=null. No fallback is used.",
    }
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
