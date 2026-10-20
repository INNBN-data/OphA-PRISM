from __future__ import annotations
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("VISION", "ACCESS", "CONTROL", "RECON", "CURE")


class ValidationError(ValueError):
    pass


def load_yaml(path):
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def configuration():
    return load_yaml(ROOT / "config/model_settings.yaml")


def fingerprint(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_value(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def binary(series):
    value = pd.to_numeric(series, errors="coerce")
    if (series.notna() & value.isna()).any() or (value.notna() & ~value.isin([0, 1])).any():
        raise ValidationError("invalid_binary_value")
    return value.astype("Float64")


def read_csv(path):
    # Read identifiers as strings, preserving leading zeros.
    import csv
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    if len(header) != len(set(header)):
        raise ValidationError("duplicate_csv_headers")
    data = pd.read_csv(path, dtype={c: "string" for c in ("patient_id", "lesion_id", "op_id") if c in header},
                       low_memory=False)
    return data.replace([-999, "-999"], np.nan)


def require_columns(data, columns):
    missing = sorted(set(columns) - set(data.columns))
    if missing:
        raise ValidationError("missing_required_columns: " + ", ".join(missing))


def identifiers(data):
    require_columns(data, ["patient_id"])
    value = data.patient_id.astype("string")
    if value.isna().any() or value.str.strip().eq("").any() or value.duplicated().any():
        raise ValidationError("patient_id_must_be_unique_and_nonmissing")
    return value.astype(str)


def private_output(path):
    """Require generated records/models to remain below the ignored private root."""
    path = Path(path).resolve()
    allowed = (ROOT / "runs/private").resolve()
    if not path.is_relative_to(allowed):
        raise ValidationError("generated_records_must_be_inside_runs/private")
    path.mkdir(parents=True, exist_ok=True)
    return path


def software_record():
    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "catboost", "xgboost", "optuna", "shap"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version.split()[0],
            "platform": platform.platform(), "packages": packages,
            "code_origin": "new reference implementation; not historical model version 1.2.0"}
