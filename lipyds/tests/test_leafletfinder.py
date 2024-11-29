import pytest

import numpy as np
from numpy.testing import assert_equal
import MDAnalysis as mda
from MDAnalysisTests.datafiles import (Martini_membrane_gro,
                                       GRO_MEMPROT, XTC_MEMPROT)
from .datafiles import (Martini_double_membrane, DPPC_vesicle_only,
                        NEURONAL_DDAT, NEURONAL_HDAT,
                        NEURONAL_HSERT, NEURONAL_GLYT2)

from lipyds import LeafletFinder

try:
    import networkx
    HAS_NX = True
except ImportError:
    HAS_NX = False

class BaseTestLeafletFinder:
    select = "name PO4"

    @pytest.fixture()
    def universe(self):
        return mda.Universe(self.file)

    def test_leafletfinder(self, universe, method, kwargs):
        lf = LeafletFinder(universe, select=self.select, pbc=True,
                           method=method, **kwargs)

        for found, given in zip(lf.leaflet_residues, self.leaflet_resix):
            assert_equal(found.resindices, given,
                         err_msg="Found wrong leaflet lipids")
            
    @pytest.mark.parametrize("update_TopologyAttr", [True, False])
    def test_updateTopologyAttr(self, universe, method, kwargs, update_TopologyAttr):
        lf = LeafletFinder(universe, select=self.select, pbc=True,
                           method=method, update_TopologyAttr=update_TopologyAttr, **kwargs)
        lf.run()
        assert lf._update_TopologyAttr == update_TopologyAttr

        if update_TopologyAttr:
            for index, leaflet_resix in enumerate(self.leaflet_resix):
                residues = universe.residues[leaflet_resix]
                leaflet_index = set(residues.leaflets)
                assert len(leaflet_index) == 1
                assert leaflet_index.pop() == index
                


class BaseTestSinglePlanar(BaseTestLeafletFinder):
    file = Martini_membrane_gro
    leaflet_resix = [np.arange(180), np.arange(225, 405)]

@pytest.mark.parametrize("method, kwargs", [
    ("spectralclustering", {"cutoff": 40}),
])
class TestSinglePlanar(BaseTestSinglePlanar):
    """Test the core clustering methods"""

@pytest.mark.skipif(not HAS_NX, reason='needs networkx')
@pytest.mark.parametrize("method, kwargs", [
    ("graph", {"cutoff": 20}),
])
class TestSinglePlanarGraph(BaseTestSinglePlanar):
    """Test the graph method"""



@pytest.mark.parametrize("method, kwargs", [
    ("spectralclustering", {"cutoff": 40, "n_leaflets": 4}),

])
class TestDoublePlanar(BaseTestLeafletFinder):
    file = Martini_double_membrane
    leaflet_resix = [np.arange(450, 630),
                     np.arange(675, 855),
                     np.arange(180),
                     np.arange(225, 405)]


class BaseTestVesicle:
    file = DPPC_vesicle_only
    select = "name PO4"
    n_leaflets = 2

    full_20 = ([0,   43,   76,  112,  141,  172,  204,
                234,  270,  301,  342,  377,  409,  441,
                474,  513,  544,  579,  621,  647,  677,
                715,  747,  771,  811,  847,  882,  914,
                951,  982, 1016, 1046, 1084, 1116, 1150,
                1181, 1210, 1246, 1278, 1312, 1351, 1375,
                1401, 1440, 1476, 1505, 1549, 1582, 1618,
                1648, 1680, 1713, 1740, 1780, 1810, 1841,
                1864, 1899, 1936, 1974, 1999, 2033, 2066,
                2095, 2127, 2181, 2207, 2243, 2278, 2311,
                2336, 2368, 2400, 2427, 2456, 2482, 2515,
                2547, 2575, 2608, 2636, 2665, 2693, 2720,
                2748, 2792, 2822, 2860, 2891, 2936, 2960,
                2992, 3017],
               [ 3,   36,   89,  139,  198,  249,  298,
                340,  388,  435,  491,  528,  583,  620,
                681,  730,  794,  831,  877,  932,  979,
                1032, 1073, 1132, 1180, 1238, 1286, 1328,
                1396, 1441, 1490, 1528, 1577, 1625, 1688,
                1742, 1782, 1839, 1910, 1945, 2005, 2057,
                2111, 2153, 2180, 2236, 2286, 2342, 2401,
                2470, 2528, 2584, 2649, 2722, 2773, 2818,
                2861, 2905, 2961])

    half_20 = ([0,   74,  134,  188,  250,  306,  362,
                452,  524,  588,  660,  736,  796,  872,
                928,  996, 1066, 1120, 1190, 1252, 1304,
                1374, 1434, 1512, 1576, 1638, 1686, 1750,
                1818, 1872, 1954, 2008, 2078, 2146, 2222,
                2296, 2346, 2398, 2460, 2524, 2590, 2646,
                2702, 2756, 2836, 2900, 2958, 3012],
               [4,   98,  228,  350,  434,  518,  614,
                696,  806,  912, 1006, 1124, 1220, 1328,
                1452, 1528, 1666, 1776, 1892, 1972, 2088,
                2174, 2264, 2410, 2520, 2626, 2766, 2854,
                2972])

    fifth_20 = ([0,  175,  355,  540,  735,  890, 1105,
                1270, 1430, 1580, 1735, 1885, 2095, 2300,
                2445, 2585, 2720, 2885, 3020],
                [5,  265,  465,  650,  915, 1095, 1325,
                    1675, 1920, 2115, 2305, 2640, 2945])

    
    @pytest.fixture()
    def universe(self):
        return mda.Universe(self.file)


class BaseTestVesicleFull(BaseTestVesicle):
    def test_full(self, universe, method, kwargs):
        lf = LeafletFinder(universe.atoms, select=self.select,
                           n_leaflets=self.n_leaflets, pbc=True,
                           method=method, **kwargs)
        
        for found, given in zip(lf.leaflet_residues, self.full_20):
            assert_equal(found.residues.resindices[::20], given,
                         err_msg="Found wrong leaflet lipids")


@pytest.mark.parametrize("method, kwargs", [
    ("spectralclustering", {"cutoff": 100, "delta": 10}),
])
class TestVesicleFull(BaseTestVesicleFull):
    """Test the core clustering methods"""

@pytest.mark.skipif(not HAS_NX, reason='needs networkx')
@pytest.mark.parametrize("method, kwargs", [
    ("graph", {"cutoff": 25}),
])
class TestVesicleFullGraph(BaseTestVesicleFull):
    """Test the graph method"""


@pytest.mark.parametrize("method, kwargs", [
    ("spectralclustering", {"cutoff": 100, "delta": 10}),
])
class TestVesicleHalf(BaseTestVesicle):
    def test_half(self, universe, method, kwargs):
        ag = universe.residues[::2].atoms
        lf = LeafletFinder(ag, select=self.select,
                           n_leaflets=self.n_leaflets,
                           method=method, **kwargs)
        
        for found, given in zip(lf.leaflet_residues, self.half_20):
            assert_equal(found.resindices[::20], given,
                         err_msg="Found wrong leaflet lipids")


@pytest.mark.parametrize("method, kwargs", [
    ("spectralclustering", {"cutoff": 100, "delta": 10}),
])
class TestVesicleFifth(BaseTestVesicle):
    def test_fifth(self, universe, method, kwargs):
        ag = universe.residues[::5].atoms
        lf = LeafletFinder(ag, select=self.select,
                           n_leaflets=self.n_leaflets, pbc=True,
                           method=method, **kwargs)
        
        for found, given in zip(lf.leaflet_residues, self.fifth_20):
            assert_equal(found.resindices[::20], given,
                         err_msg="Found wrong leaflet lipids")