# Data governance

Institutional records, linkage identifiers, individual probabilities, prediction sets, fitted models, preprocessing objects, and per-patient explanations remain within the controlled server environment. Exports are limited to institutionally approved metadata and aggregate summaries.

The server audit produces configured metadata and aggregate checks. Output review includes small-cell disclosure and accidental inclusion of free text. Pseudonymous identifiers remain patient-level linkage fields.

Generated analytical material is stored under `runs/private/`. The public archive is built from `RELEASE_FILES.txt`; new CSVs and fitted objects are not included automatically. The distributed CSVs contain only metadata or an empty schema.

De-identified data access may be considered upon reasonable request, subject to institutional approval and an appropriate agreement. The MIT license applies to code and documentation; it does not grant access to clinical records. Publication of fitted or preprocessing objects requires separate review because those objects may retain training information.
