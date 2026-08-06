# ADR 0001: Use a `src/` package layout

**Status:** accepted.

Separating importable code from the repository root prevents tests from accidentally importing an uninstalled working tree and makes package builds representative. Tests, configs, data, scripts, and docs remain first-class top-level concerns.

