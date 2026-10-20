"""Separate code-package readiness from verified numerical reproducibility."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repository_audit import audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-verified-results", action="store_true")
    args = parser.parse_args()
    result = audit()
    result["empirical_requirements"] = [
        "Actual server dictionary/142-feature list and saved fold/prediction files reconciled.",
        "Held-out imputation, probability/conformal partitions and repeated-set rule verified against actual code.",
        "Sample-size console output and agreement estimator denominators reconciled.",
        "All reported clinical metrics, confidence intervals and figure source tables checked against server outputs."]
    print(json.dumps(result, indent=2))
    if result["status"] != "passed" or args.require_verified_results:
        sys.exit(2)
