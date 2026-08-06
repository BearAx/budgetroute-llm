# Retrieval

The default retrieval system is local, deterministic, and portable.

1. Load UTF-8 `.txt`, `.md`, or typed `.jsonl` documents in sorted path order.
2. Collapse whitespace and create overlapping, deterministic character chunks with document/chunk IDs and provenance metadata.
3. Embed each chunk. Fake mode uses signed hashing of lexical tokens; the optional Transformers provider mean-pools attention-masked hidden states.
4. Normalize vectors and persist them with typed chunk metadata in a compressed NumPy archive.
5. Embed and normalize the query, compute exact cosine dot products, perform stable top-k ranking, and apply the minimum-score threshold.
6. Record retrieval time separately and pass selected chunks to generation with IDs and scores.

Fake embeddings are collision-prone lexical fixtures, not semantic representations. Exact NumPy search is appropriate for small local corpora but scales linearly. Setting `index_type: faiss` selects an optional `IndexFlatIP` implementation while retaining portable NumPy metadata persistence; install the `retrieval` extra on a supported platform. FAISS is not required on Windows or in default tests.

All non-original corpora need provenance and license notes. Built indexes and downloads are ignored by Git.
