# Retrieval

The default retrieval system is local, deterministic, and portable.

1. Load UTF-8 `.txt`, `.md`, or typed `.jsonl` documents in sorted path order.
2. Collapse whitespace and create overlapping, deterministic character chunks with document/chunk IDs and provenance metadata.
3. Embed each chunk. Fake mode uses signed hashing of lexical tokens; the optional Transformers provider mean-pools attention-masked hidden states from an exact configured model revision.
4. Normalize vectors and persist them with typed chunk metadata in a compressed NumPy archive.
5. Embed and normalize the query, perform exact or configured approximate inner-product search, and apply the minimum-score threshold.
6. Record retrieval time separately and pass selected chunks to generation with IDs and scores.

Fake embeddings are collision-prone lexical fixtures, not semantic representations. Exact NumPy search is appropriate for small local corpora but scales linearly. `index_type: faiss` selects exact `IndexFlatIP`; `faiss_hnsw` selects `IndexHNSWFlat` with explicit neighbor, construction-ef, and search-ef settings. Both retain portable NumPy vectors/chunk metadata, and HNSW is rebuilt on load. Install the `retrieval` extra on a supported platform; FAISS is not required in default tests.

`configs/retrieval/semantic-hnsw.yaml` pins `sentence-transformers/all-MiniLM-L6-v2` to an exact upstream revision and declares its Apache-2.0 metadata. This makes the implementation reproducible, not universally high quality. Measure approximate recall against exact search, end-to-end answer benefit, memory, and latency on the actual corpus before choosing HNSW parameters.

The `learned_retrieval` policy uses paired benchmark outcomes to estimate whether retrieval improves quality. A similarity threshold and a benefit classifier remain fallible signals; corpus/model/prompt changes invalidate prior calibration.

All non-original corpora need provenance and license notes. Built indexes and downloads are ignored by Git.
