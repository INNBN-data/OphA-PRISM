"""Check the public reference package; this does not validate clinical results."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
METADATA_CSV = {"data/confirmed_variable_subset.csv", "data/registry_schema.csv", "data/missingness_codes.csv"}
METADATA_JSON = {"server_audit/audit_config.json"}
ALLOWED_SUFFIXES = {".py", ".md", ".txt", ".yaml", ".yml", ".toml", ".cff", ".R", ".slurm", ".csv"}
ROOT_FILES = {"LICENSE", "VERSION", "Dockerfile", ".gitignore", "RELEASE_FILES.txt", "MANIFEST.sha256"}


def release_path_allowed(relative):
    p = Path(relative)
    if p.is_absolute() or ".." in p.parts or "__pycache__" in p.parts:
        return False
    if p.parts and p.parts[0] in ("runs", "upload", "outputs", "logs"):
        return False
    if relative.startswith("data/private/"):
        return relative == "data/private/README.md"
    if p.suffix == ".csv":
        return relative in METADATA_CSV
    if p.suffix == ".json":
        return relative in METADATA_JSON
    if p.suffix not in ALLOWED_SUFFIXES and relative not in ROOT_FILES:
        return False
    return True


def selected_files():
    listed = (ROOT / "RELEASE_FILES.txt").read_text(encoding="utf-8").splitlines()
    listed = [line.strip() for line in listed if line.strip() and not line.startswith("#")]
    if len(listed) != len(set(listed)):
        raise ValueError("duplicate_release_paths")
    return listed


def audit(check_hashes=True):
    errors = []
    required = ["README.md", "LICENSE", "CITATION.cff", "VERSION", "run_pipeline.py",
                "config/model_settings.yaml", "docs/REPRODUCIBILITY.md", "docs/RELEASE_STATUS.md",
                "results/author_reported_metrics.yaml", "server_audit/opha_prism_server_audit.py"]
    for relative in required:
        if not (ROOT / relative).is_file():
            errors.append("missing_required_file: " + relative)
    selected = selected_files()
    for relative in selected:
        path = ROOT / relative
        if not release_path_allowed(relative) or not path.resolve().is_relative_to(ROOT) or path.is_symlink():
            errors.append("unsafe_public_release_path: " + relative)
            continue
        if not path.is_file():
            errors.append("missing_allowlisted_file: " + relative)
            continue
        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=relative)
            except SyntaxError:
                errors.append("python_syntax_error: " + relative)
    with (ROOT / "data/registry_schema.csv").open() as handle:
        if sum(1 for _ in csv.reader(handle)) != 1:
            errors.append("schema_csv_must_have_no_patient_rows")
    if check_hashes and (ROOT / "MANIFEST.sha256").is_file():
        expected = {}
        for line in (ROOT / "MANIFEST.sha256").read_text().splitlines():
            sha, relative = line.split(None, 1)
            expected[relative] = sha
        for relative in selected:
            path = ROOT / relative
            if path.is_file() and expected.get(relative) != hashlib.sha256(path.read_bytes()).hexdigest():
                errors.append("manifest_mismatch: " + relative)
        if set(expected) != set(selected):
            errors.append("manifest_and_release_allowlist_differ")
    return {"status": "passed" if not errors else "failed", "errors": errors,
            "public_files": len(selected), "scope": "Reference software structure, syntax, hashes and release path policy.",
            "historical_results_verified": False,
            "clinical_release_status": "Server-based empirical reconciliation remains required."}


if __name__ == "__main__":
    result = audit()
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["status"] == "passed" else 2)
