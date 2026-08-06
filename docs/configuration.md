# Configuration

`pyproject.toml` is authoritative for packaging and tool settings. Runtime behavior uses typed Pydantic models loaded from YAML.

## Composition

A file may contain `extends` as one path or an ordered path list. Paths resolve relative to the including file first, then the current working directory. Included mappings are deep-merged left to right; the current file wins. Cycles, missing files, non-mapping roots, and invalid types fail with actionable errors.

Model identifiers live only in `configs/models/`. Routing fragments live in `configs/routing/`. Benchmark and serving files compose base/model values and add workload-specific settings.

## Validation

Validation rejects missing Transformers model IDs, fake quantization, CPU FP16, CPU quantization, invalid retrieval chunk overlap, missing real embedding IDs, unusable random probabilities, learned routing without an artifact path, retrieval-first without retrieval, and fake mode with real backends.

Runtime initialization separately rejects explicit CUDA or BF16 requests unsupported by the installed PyTorch/hardware and missing quantization dependencies. Explicit GPU selection never silently falls back. `device: auto` may select CPU or CUDA and records the result.

## Environment overrides

Operational values can be overridden with:

```text
BUDGETROUTE_OUTPUT_DIR
BUDGETROUTE_HOST
BUDGETROUTE_PORT
BUDGETROUTE_LOG_LEVEL
```

The API configuration response is sanitized and does not include failure markers, credentials, or arbitrary environment variables. The project does not load `.env` implicitly; orchestration may provide these variables.

