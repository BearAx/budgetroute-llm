# Contributing

Create an issue before a substantial change and use a focused branch. Bootstrap with `scripts/bootstrap.ps1` on Windows or `scripts/bootstrap.sh` on Unix. Multi-module changes require an execution plan as described in `PLANS.md`.

Run formatting, linting, MyPy, unit tests, integration tests, and package build through `scripts/check.ps1` or `scripts/check.sh`. Add tests for behavior changes and update public documentation and configuration examples.

Default tests must remain deterministic and offline. Do not commit credentials, model weights, downloaded datasets, indexes, caches, or ordinary outputs. Never report fake backend timings or quality as real performance, and never add benchmark claims without preserving the underlying artifacts and environment metadata.

Pull requests should explain the user-visible change, methodology impact, validation performed, and remaining limitations. Keep commits reviewable; signed commits are not required.

