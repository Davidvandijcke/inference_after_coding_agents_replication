"""Rerun the single chosen analysis: effect of Pathways to Work on employment
at 12 months (employed_m12), OLS on treatment + site indicators, HC2 SEs,
two-sided t-test on the treatment coefficient (CODEBOOK.md convention)."""
import argparse, json
import pandas as pd
import statsmodels.formula.api as smf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.data)
    r = smf.ols("employed_m12 ~ treatment + C(site)", d).fit(cov_type="HC2", use_t=True)
    out = {"estimate": float(r.params["treatment"]),
           "std_error": float(r.bse["treatment"]),
           "p_value": float(r.pvalues["treatment"])}
    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
