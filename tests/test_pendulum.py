import pytest

from f1sim.pendulum import check


@pytest.mark.slow
def test_tqc_learns_the_pendulum():
    assert check(log=None) > -250  # random: about -1,200
