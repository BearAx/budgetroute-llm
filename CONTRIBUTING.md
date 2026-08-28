# Contributing

Create an issue before a substantial change and use a focused branch. Bootstrap with `scripts/bootstrap.ps1` on Windows or `scripts/bootstrap.sh` on Unix. Explain multi-module architecture, migration, validation, and rollback considerations in the issue or pull request.

Run formatting, linting, MyPy, unit tests, integration tests, and package build through `scripts/check.ps1` or `scripts/check.sh`. Add tests for behavior changes and update public documentation and configuration examples.

Default tests must remain deterministic and offline. The isolated PostgreSQL contract runs only when `BUDGETROUTE_TEST_POSTGRES_DSN` names a disposable test database. Do not commit credentials, DSNs, model weights, downloaded datasets, indexes, caches, operational databases, calibration registries, or ordinary outputs. Keep runtime secrets in environment-backed test fixtures and assert that sanitized responses/artifacts omit them. Never edit an applied migration; add a forward-only numbered file and document rolling compatibility. Never load untrusted `joblib` routing artifacts. Never report fake backend timings or quality as real performance, and never add batching/runtime/quantization speed claims without preserving the underlying artifacts and environment metadata.

Pull requests should explain the user-visible change, methodology impact, validation performed, and remaining limitations. Keep commits reviewable; signed commits are not required.
