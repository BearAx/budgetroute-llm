from __future__ import annotations

from budgetroute.api.security import opaque_identity


def test_opaque_identity_is_stable_and_hides_the_token() -> None:
    token = "test-only-low-entropy-token"

    identity = opaque_identity(token, "127.0.0.1")

    assert identity == opaque_identity(token, "198.51.100.2")
    assert token not in identity
    assert len(identity) == 64
    assert identity != opaque_identity("another-token", "127.0.0.1")


def test_opaque_identity_separates_host_fallback_from_token_namespace() -> None:
    assert opaque_identity(None, "client") != opaque_identity("host:client", None)
