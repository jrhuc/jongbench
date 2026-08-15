from __future__ import annotations

import hashlib
import io

import pytest

from jongbench import weights


def test_custom_checkpoint_must_exist(tmp_path) -> None:
    checkpoint = tmp_path / "custom.pth"
    checkpoint.write_bytes(b"custom")
    assert weights.resolve_mortal_weights(checkpoint) == checkpoint

    with pytest.raises(FileNotFoundError, match="Mortal checkpoint not found"):
        weights.resolve_mortal_weights(tmp_path / "missing.pth")


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

    path = weights.resolve_mortal_weights()
    assert path.read_bytes() == payload
    assert weights.resolve_mortal_weights() == path
    assert calls == 1


def test_auto_checkpoint_can_use_a_verified_environment_source(
    tmp_path, monkeypatch
) -> None:
    payload = b"reviewer checkpoint"
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setenv("JONGBENCH_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("JONGBENCH_WEIGHTS_URL", "https://example.test/reviewer.pth")
    monkeypatch.setenv("JONGBENCH_WEIGHTS_SHA256", digest.upper())

    def open_checkpoint(request, timeout):
        assert request.full_url == "https://example.test/reviewer.pth"
        assert timeout == 60
        return io.BytesIO(payload)

    monkeypatch.setattr(weights, "urlopen", open_checkpoint)

    path = weights.resolve_mortal_weights()
    assert path == tmp_path / "jongbench" / f"weights-{digest[:12]}.pth"
    assert path.read_bytes() == payload
    assert weights.auto_weights_sha256() == digest


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
