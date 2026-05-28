import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests that download model weights or run full inference (deselect with -m 'not slow')",
    )
