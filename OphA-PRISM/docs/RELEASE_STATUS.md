# Release status

**Release:** 1.0.1, 20 October 2026. **Classification:** reference research software.

| Component | Verification status |
| --- | --- |
| Public source structure, syntax, and checksums | Checked locally |
| Software tests | Recorded in SOFTWARE_TEST_REPORT.md |
| Clinical data in the archive | None |
| Complete 408-variable dictionary and ordered 142-feature list | Required server inputs; not inspected |
| Historical v1.2.0 code, fitted models, and training histories | Not verified |
| Clinical OOF, LOEO, and reserved-cohort predictions | Not verified |
| Manuscript metrics and empirical figures | Not reproduced |
| Original R environment and sample-size calculation | Server execution required |
| Docker image | Definition supplied; build not tested |

The archive is suitable for public distribution as reference software. Exact reproduction of the manuscript requires the server evidence in [IMPLEMENTATION_VERIFICATION.md](../server_audit/IMPLEMENTATION_VERIFICATION.md). Software checks establish the tested behavior of the implementation; they do not establish clinical validity or suitability for treatment selection.
