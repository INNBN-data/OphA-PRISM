"""Build a code-only public ZIP from an explicit allowlist, never a recursive dump."""
from pathlib import Path
import hashlib
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repository_audit import audit, selected_files, release_path_allowed


def main():
    result = audit(check_hashes=False)
    if result["status"] != "passed":
        raise SystemExit(json.dumps(result, indent=2))
    selected = selected_files()
    manifest = []
    for relative in selected:
        if not release_path_allowed(relative):
            raise SystemExit("Unsafe release path refused.")
        manifest.append(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() + "  " + relative)
    (ROOT / "MANIFEST.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    destination = ROOT.parent / ("OphA_PRISM_reference_v" + (ROOT / "VERSION").read_text().strip() + ".zip")
    temporary = destination.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in selected + ["MANIFEST.sha256"]:
            archive.write(ROOT / relative, arcname="OphA-PRISM/" + relative)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise SystemExit("ZIP validation failed.")
    temporary.replace(destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(".zip.sha256").write_text(digest + "  " + destination.name + "\n")
    print(json.dumps({"archive": str(destination), "sha256": digest, "public_files": len(selected) + 1,
                      "status": "reference_code_release_built", "clinical_results_reproduced": False}, indent=2))


if __name__ == "__main__":
    main()
