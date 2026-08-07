# Development data

`sample_benchmark.jsonl` and `sample_corpus/` are small, original development fixtures for exercising routing, evaluation, retrieval, and artifact generation. They contain no personal or proprietary data and are released under the repository MIT license.

They are not representative scientific benchmarks. Revision-pinned GSM8K, MMLU, and HotpotQA contracts are provided under `configs/datasets/`. Materialize them with `budgetroute materialize-dataset`; every output receives a sibling hash/license/source manifest, and HotpotQA can emit its retrieval corpus.

Downloaded data, materialized benchmark JSONL, extracted corpora, and built indexes belong in ignored `data/downloads/`, `data/materialized/`, `data/corpora/`, and `data/indexes/` directories. Do not commit upstream datasets without independently confirming redistribution terms.
