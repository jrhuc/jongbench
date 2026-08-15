"""PEP 517 wrapper that enforces complete source and wheel artifacts."""

from pathlib import Path
from typing import Any

from setuptools import build_meta as _setuptools_backend
from setuptools.build_meta import *  # noqa: F403


_WEBUI_PAGE = Path(__file__).resolve().parent / "jongbench" / "webui_page.html"
_WEBUI_BUILD = "cd webui && bun install --frozen-lockfile && bun run build"


def _require_webui_page() -> None:
    if not _WEBUI_PAGE.is_file() or _WEBUI_PAGE.stat().st_size == 0:
        raise RuntimeError(
            "jongbench/webui_page.html is required in every distribution. "
            "Restore the versioned artifact or regenerate it with "
            f"`{_WEBUI_BUILD}` before building."
        )


def build_sdist(sdist_directory: str, config_settings: dict[str, Any] | None = None) -> str:
    _require_webui_page()
    return _setuptools_backend.build_sdist(sdist_directory, config_settings)


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    _require_webui_page()
    return _setuptools_backend.build_wheel(
        wheel_directory, config_settings, metadata_directory
    )


def build_editable(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    _require_webui_page()
    return _setuptools_backend.build_editable(
        wheel_directory, config_settings, metadata_directory
    )
