# Public upload

1. Extract the release archive and upload the **contents of `OphA-PRISM/`** to the repository root. Preserve directory paths and include `.github/` and `.gitignore`.
2. Include `LICENSE`, `CITATION.cff`, `VERSION`, `RELEASE_FILES.txt`, and `MANIFEST.sha256`.
3. Run the software tests and `python repository_audit.py` in the installed environment.
4. Publish the release as reference research software, with the scope stated in `README.md` and `docs/RELEASE_STATUS.md`.
5. Add the actual repository URL and archival DOI to the citation metadata after assignment. Regenerate the manifest and archive after any file changes.

Private run directories, clinical records, individual predictions, and fitted objects are excluded from this release. Only separately approved aggregate additions may be included in a later archive.

