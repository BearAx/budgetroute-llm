from __future__ import annotations

from budgetroute.api.security import opaque_client_identity


def test_opaque_client_identity_is_stable_and_hides_the_host() -> None:
    host = "198.51.100.2"

    identity = opaque_client_identity(host)

    assert identity == opaque_client_identity(host)
    assert host not in identity
    assert len(identity) == 64
    assert identity != opaque_client_identity("203.0.113.7")


def test_opaque_client_identity_has_an_unknown_host_fallback() -> None:
    assert opaque_client_identity(None) == opaque_client_identity(None)
    assert opaque_client_identity(None) != opaque_client_identity("unknown")
