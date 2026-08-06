# ADR 0002: Isolate generation behind a protocol

**Status:** accepted.

Fake and Transformers implementations share initialize, health, token count, single/batch generation, metadata, and cleanup. Routing and evaluation depend on structured generations rather than framework objects. This keeps default imports/tests offline and allows future runtimes without rewriting orchestration.

