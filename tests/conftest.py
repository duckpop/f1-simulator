import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import pytest


@pytest.fixture(scope="session")
def tracks():
    from f1env import generatemap

    return [generatemap(seed=s) for s in range(2)]


@pytest.fixture(scope="session")
def bank(tracks):
    from f1env.sensors import make_bank

    return make_bank(tracks)
