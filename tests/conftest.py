"""Pytest configuration and fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def reset_model_cache():
    """Reset the model cache between tests."""
    from rapidocr_service.model import get_model

    get_model.cache_clear()
    yield
    get_model.cache_clear()
