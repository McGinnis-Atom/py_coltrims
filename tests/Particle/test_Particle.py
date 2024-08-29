import coltrims


def almostEqual(a, b):
    import numpy as np
    return np.abs(a - b) < 1e-3

class TestColtrims:
    def test_t1(self):
        assert 1==1
class TestColtrims2:
    def test_t2(self):
        assert 2==2