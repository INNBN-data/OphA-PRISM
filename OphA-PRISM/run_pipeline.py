#!/usr/bin/env python3
"""OphA-PRISM reference analysis command line."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="Check public repository structure, manifests and release policy.")
    p = sub.add_parser("audit", help="Run the aggregate-only server audit on existing files.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", required=True, help="New aggregate JSON path; existing files are not overwritten.")
    p = sub.add_parser("validate", help="Validate the real private cohorts and complete dictionary.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--include-holdout", action="store_true")
    for name in ("train", "collect", "fit-final", "holdout", "temporal", "collect-temporal", "diagnostic", "diagnostic-summary", "baselines"):
        p = sub.add_parser(name)
        p.add_argument("--task", choices=["VISION", "ACCESS", "CONTROL", "RECON", "CURE"], required=True)
        p.add_argument("--run", required=True, help="New run folder name below runs/private.")
        if name in ("train", "fit-final", "holdout", "temporal", "diagnostic", "baselines"):
            p.add_argument("--data-root", required=True)
        if name in ("train", "baselines"):
            p.add_argument("--repeat", type=int, required=True)
            p.add_argument("--fold", type=int, required=True)
        if name == "train":
            p.add_argument("--resume", action="store_true")
        if name == "collect":
            p.add_argument("--bootstrap", type=int, default=2000)
        if name == "temporal":
            p.add_argument("--test-era", type=int, choices=[1, 2, 3], required=True)
        if name in ("temporal", "collect-temporal"):
            p.add_argument("--rolling", action="store_true")
        if name in ("diagnostic", "diagnostic-summary"):
            p.add_argument("--kind", choices=["bootstrap", "permutation"], required=True)
        if name == "diagnostic":
            p.add_argument("--index", type=int, required=True)
        if name == "diagnostic-summary":
            p.add_argument("--expected", type=int, required=True)
    p = sub.add_parser("figures", help="Generate empirical panels from a unique-patient probability file.")
    p.add_argument("--source", required=True)
    p.add_argument("--task", choices=["VISION", "ACCESS", "CONTROL", "RECON", "CURE"], required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    return args


def execute(args):
    from ophaprism.common import ROOT, ValidationError, json_value
    if args.command == "check":
        from repository_audit import audit
        result = audit()
    elif args.command == "audit":
        return subprocess.run([sys.executable, str(ROOT / "server_audit/opha_prism_server_audit.py"),
                               "--root", args.data_root, "--output", args.output], check=False).returncode
    elif args.command == "validate":
        from ophaprism.data import validate_inputs
        result = validate_inputs(args.data_root, args.include_holdout)
    else:
        if not args.run or Path(args.run).name != args.run or args.run in (".", ".."):
            raise ValidationError("run_must_be_one_folder_name")
        output = ROOT / "runs/private" / args.run
        if args.command in ("train", "collect", "fit-final", "holdout", "temporal", "collect-temporal"):
            from ophaprism import workflow
            if args.command == "train":
                result = workflow.train_fold(args.data_root, output, args.task, args.repeat, args.fold, resume=args.resume)
            elif args.command == "collect":
                result = workflow.collect(output, args.task, args.bootstrap)
            elif args.command == "fit-final":
                result = workflow.fit_final(args.data_root, output, args.task)
            elif args.command == "holdout":
                result = workflow.evaluate_holdout(args.data_root, output, args.task)
            elif args.command == "temporal":
                result = workflow.temporal_fit(args.data_root, output, args.task, args.test_era, args.rolling)
            else:
                result = workflow.collect_temporal(output, args.task, args.rolling)
        elif args.command in ("diagnostic", "diagnostic-summary"):
            from ophaprism import diagnostics
            if args.command == "diagnostic":
                result = diagnostics.diagnostic_replicate(args.data_root, output, args.task, args.kind, args.index)
            else:
                result = diagnostics.summarize_replicates(output, args.task, args.kind, args.expected)
        elif args.command == "baselines":
            if args.task != "CURE":
                raise ValidationError("baseline_comparison_here_is_for_cure")
            from ophaprism.baselines import comparisons
            result = comparisons(args.data_root, output, args.repeat, args.fold)
        elif args.command == "figures":
            from ophaprism.figures import plot_predictions
            plot_predictions(args.source, output / args.task / "figures", args.task, args.bootstrap)
            result = {"status": "empirical_diagnostic_panels_created", "task": args.task}
        else:
            raise ValidationError("unknown_command")
    print(json.dumps(json_value(result), ensure_ascii=False, indent=2, allow_nan=False))
    if isinstance(result, dict) and (result.get("status") == "failed" or str(result.get("status", "")).startswith("not_estimable")):
        return 2
    return 0


if __name__ == "__main__":
    try:
        sys.exit(execute(main()))
    except Exception as exc:
        # Do not expose row values or patient identities in a public terminal log.
        from ophaprism.common import ValidationError
        if isinstance(exc, ValidationError):
            print("Analysis stopped: " + str(exc), file=sys.stderr)
        else:
            print("Analysis stopped: " + type(exc).__name__ + "; inspect the controlled server environment.", file=sys.stderr)
        sys.exit(2)
