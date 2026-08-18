from __future__ import annotations

import hashlib
import io
from dataclasses import FrozenInstanceError

import pytest

from jongbench import weights


@pytest.fixture(autouse=True)
def _clean_weights_environment(monkeypatch) -> None:
    for name in (
        weights.WEIGHTS_URL_ENV,
        weights.WEIGHTS_SHA256_ENV,
        weights.WEIGHTS_USE_POLICY_ENV,
        "JONGBENCH_CACHE_DIR",
        "XDG_CACHE_HOME",
    ):
        monkeypatch.delenv(name, raising=False)


def test_custom_checkpoint_has_an_immutable_resolved_identity(tmp_path) -> None:
    checkpoint = tmp_path / "custom.pth"
    checkpoint.write_bytes(b"custom")

    resolved = weights.resolve_mortal_checkpoint(checkpoint, use_policy=True)

    assert resolved.path == checkpoint
    assert resolved.sha256 == hashlib.sha256(b"custom").hexdigest()
    assert resolved.source == str(checkpoint)
    assert resolved.use_policy is True
    assert resolved.as_dict() == {
        "path": str(checkpoint),
        "sha256": resolved.sha256,
        "source": str(checkpoint),
        "use_policy": True,
    }
    assert weights.resolve_mortal_checkpoint(resolved) is resolved
    assert weights.resolve_mortal_weights(resolved) == checkpoint
    with pytest.raises(FrozenInstanceError):
        resolved.use_policy = False  # type: ignore[misc]


def test_custom_checkpoint_must_exist(tmp_path) -> None:
    checkpoint = tmp_path / "custom.pth"
    checkpoint.write_bytes(b"custom")
    assert weights.resolve_mortal_weights(checkpoint) == checkpoint

    with pytest.raises(FileNotFoundError, match="Mortal checkpoint not found"):
        weights.resolve_mortal_weights(tmp_path / "missing.pth")


def test_explicit_policy_overrides_invalid_ambient_value(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "custom.pth"
    checkpoint.write_bytes(b"custom")
    monkeypatch.setenv(weights.WEIGHTS_USE_POLICY_ENV, "not-a-boolean")

    resolved = weights.resolve_mortal_checkpoint(checkpoint, use_policy=False)

    assert resolved.use_policy is False
    with pytest.raises(ValueError, match=weights.WEIGHTS_USE_POLICY_ENV):
        weights.resolve_mortal_checkpoint(checkpoint)


def test_auto_checkpoint_is_downloaded_verified_and_cached(
    tmp_path, monkeypatch
) -> None:
    payload = b"verified checkpoint"
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setenv("JONGBENCH_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(weights, "MORTAL_WEIGHTS_SHA256", digest)
    monkeypatch.setattr(weights, "MORTAL_WEIGHTS_FILENAME", "mortal-test.pth")
    calls = 0

    def open_checkpoint(request, timeout):
        nonlocal calls
        calls += 1
        assert request.full_url == weights.MORTAL_WEIGHTS_URL
        assert timeout == 60
        return io.BytesIO(payload)

    monkeypatch.setattr(weights, "urlopen", open_checkpoint)

    resolved = weights.resolve_mortal_checkpoint(use_policy=False)
    assert resolved.path.read_bytes() == payload
    assert resolved.sha256 == digest
    assert resolved.source == weights.MORTAL_WEIGHTS_URL
    assert resolved.use_policy is False
    assert weights.resolve_mortal_weights() == resolved.path
    assert calls == 1


def test_auto_checkpoint_can_use_a_verified_environment_source(
    tmp_path, monkeypatch
) -> None:
    payload = b"reviewer checkpoint"
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setenv("JONGBENCH_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("JONGBENCH_WEIGHTS_URL", "https://example.test/reviewer.pth")
    monkeypatch.setenv("JONGBENCH_WEIGHTS_SHA256", digest.upper())
    monkeypatch.setenv("JONGBENCH_WEIGHTS_USE_POLICY", "true")

    def open_checkpoint(request, timeout):
        assert request.full_url == "https://example.test/reviewer.pth"
        assert timeout == 60
        return io.BytesIO(payload)

    monkeypatch.setattr(weights, "urlopen", open_checkpoint)

    resolved = weights.resolve_mortal_checkpoint()
    assert resolved.path == tmp_path / "jongbench" / f"weights-{digest[:12]}.pth"
    assert resolved.path.read_bytes() == payload
    assert resolved.sha256 == digest
    assert resolved.source == "https://example.test/reviewer.pth"
    assert resolved.use_policy is True
    assert weights.auto_weights_sha256() == digest


def test_resolved_checkpoint_does_not_reread_ambient_config(
    tmp_path, monkeypatch
) -> None:
    checkpoint = tmp_path / "custom.pth"
    checkpoint.write_bytes(b"custom")
    resolved = weights.resolve_mortal_checkpoint(checkpoint, use_policy=False)
    monkeypatch.setenv(weights.WEIGHTS_USE_POLICY_ENV, "invalid")
    monkeypatch.setenv(weights.WEIGHTS_URL_ENV, "https://incomplete.test/model.pth")

    assert weights.resolve_mortal_checkpoint(resolved) is resolved


def test_auto_checkpoint_environment_source_requires_url_and_digest(
    monkeypatch,
) -> None:
    monkeypatch.setenv("JONGBENCH_WEIGHTS_URL", "https://example.test/reviewer.pth")
    with pytest.raises(ValueError, match="must be set together"):
        weights.resolve_mortal_weights()


def test_bad_checkpoint_is_not_cached(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("JONGBENCH_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(weights, "MORTAL_WEIGHTS_SHA256", "0" * 64)
    monkeypatch.setattr(weights, "MORTAL_WEIGHTS_FILENAME", "mortal-test.pth")
    monkeypatch.setattr(weights, "urlopen", lambda request, timeout: io.BytesIO(b"bad"))

    with pytest.raises(ValueError, match="checksum mismatch"):
        weights.resolve_mortal_weights()

    assert not (tmp_path / "jongbench" / "mortal-test.pth").exists()
    assert list((tmp_path / "jongbench").glob("*.tmp")) == []
