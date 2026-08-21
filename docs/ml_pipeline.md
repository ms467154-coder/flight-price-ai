# ML Pipeline

The release pipeline validates the canonical dataset, fingerprints provenance, trains baseline and candidate regressors, evaluates held-out performance, writes error analysis, records package versions and feature schema, and emits checksum-backed artifacts. The application loads the manifest at runtime and invokes the pinned model through a Python subprocess.

The current repository includes release artifacts, local tests, and a CI workflow definition. Scheduled drift monitoring, model registry promotion, and automated production rollback remain future integrations rather than claims about the current deployment.
