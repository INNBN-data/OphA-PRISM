"""Compute author-supplied developmental integer scores; no risk mapping is assumed."""
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ophaprism.common import read_csv, private_output
from ophaprism.scores import integers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--task", choices=["VISION", "CURE"], required=True)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in (".", ".."):
        raise SystemExit("Run must be a single folder name.")
    data = read_csv(args.source)
    score = integers(data, args.task)
    columns = [c for c in ("patient_id",) if c in data]
    output = data[columns].copy()
    output["developmental_integer_score"] = score
    directory = private_output(ROOT / "runs/private" / args.run / args.task / "scores")
    output.to_csv(directory / "scores.csv", index=False)
    print(json.dumps({"task": args.task, "n": len(score), "probability_mapping": "not_assumed",
                      "interpretation": "developmental score architecture; not a validated bedside tool"}))
