
import pytest

import numpy as np
from lipyds.analysis.surface import Surface
from lipyds.tests.datafiles import SURFACE_POINTS


class TestSurface:

    @pytest.fixture()
    def surface(self):
        points = np.loadtxt(SURFACE_POINTS)
        box = [309.7038, 309.7038, 140.6989, 90, 90, 90]
        return Surface(
            np.array(points),
            other_points=[],
            cutoff_other=5,
            box=np.array(box),
            cutoff=30,
            normal=[0, 0, 1],
            analysis_indices=np.arange(len(points)),
        )
    
    def test_surface(self, surface):
        assert surface is not None