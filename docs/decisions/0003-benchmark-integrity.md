# ADR 0003: Generate reports only from run artifacts

**Status:** accepted.

Benchmark execution writes resolved inputs, environment/Git identity, row-level outputs, failures, and aggregate metrics before reporting. Report code never executes a model or invents absent values. Fake artifacts carry an explicit label in metadata and Markdown so workflow validation cannot be confused with performance evidence.

