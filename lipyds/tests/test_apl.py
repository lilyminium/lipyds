import pytest

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
import MDAnalysis as mda
from .datafiles import (NEURONAL_DDAT, NEURONAL_HDAT)

from lipyds import LeafletFinder, AreaPerLipid
# from lipyds.analysis.apl import lipid_area

# @pytest.mark.parametrize("x", [0, 33, -2])
# def test_lipid_area_nopbc_x(x):
#     hg = np.array([x, 0, 0], dtype=float)
#     neighbors = np.array([[x, 1, 1],
#                           [x, 1, -1],
#                           [x, -1, 1],
#                           [x, -1, -1]], dtype=float)
#     area = lipid_area(hg, neighbors, plot=False)
#     assert_almost_equal(area, 2)

# @pytest.mark.parametrize("y", [0, 33, -2])
# def test_lipid_area_nopbc_y(y):
#     hg = np.array([0, y, 0], dtype=float)
#     neighbors = np.array([[1, y, 1],
#                           [1, y, -1],
#                           [2, y, -1],
#                           [3, y, -1],
#                           [-1, y, 1],
#                           [-1, y, -1]], dtype=float)
#     area = lipid_area(hg, neighbors, plot=False)
#     assert_almost_equal(area, 2)


# @pytest.mark.parametrize("cutoff, max_neighbors", [
#     (50, 100),
#     (80, 200),
#     (40, 80),
#     (40, 50),
# ])
# @pytest.mark.parametrize("filename, upper_popc, lower_popc", [
#     (NEURONAL_DDAT, 50, 48),
#     (NEURONAL_HDAT, 51, 49),
# ])
# class TestAPL:

#     @pytest.fixture()
#     def universe(self, filename):
#         return mda.Universe(filename)

#     @pytest.fixture()
#     def leafletfinder(self, universe):
#         lf = LeafletFinder(universe, select="name PO4", method="spectralclustering",
#                            cutoff=40, delta=6, pbc=True, n_leaflets=2)
#         return lf

#     def test_apl(self, universe, leafletfinder, upper_popc, lower_popc,
#                  cutoff, max_neighbors):
#         apl = AreaPerLipid(universe, select="name PO4 GL1 GL2 AM1 AM2 ROH",
#                            select_other="protein", cutoff=cutoff,
#                            cutoff_other=15,
#                            leafletfinder=leafletfinder,
#                            max_neighbors=max_neighbors,
#                            group_by_attr="resnames")
#         apl.run(stop=1)
#         assert_almost_equal(np.mean(apl.areas_by_attr[0]["POPC"]), lower_popc, decimal=0)
#         assert_almost_equal(np.mean(apl.areas_by_attr[1]["POPC"]), upper_popc, decimal=0)
