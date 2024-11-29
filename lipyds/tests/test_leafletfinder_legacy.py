# Tests copied from MDAnalysis
#
import MDAnalysis
import pytest

from numpy.testing import assert_equal, assert_almost_equal
import numpy as np

from MDAnalysis.tests.datafiles import Martini_membrane_gro
from lipyds.leafletfinder import LeafletFinder

try:
    import networkx
    HAS_NX = True
except ImportError:
    HAS_NX = False

@pytest.fixture()
def universe():
    return MDAnalysis.Universe(Martini_membrane_gro)


@pytest.fixture()
def lipid_heads(universe):
    return universe.select_atoms("name PO4")


@pytest.mark.skipif(HAS_NX, reason="networkx is installed")
def test_optional_nx(universe):
    errmsg = "networkx is required to use this method"
    with pytest.raises(ImportError, match=errmsg):
        LeafletFinder(
            universe,
            select="name PO4",
            pbc=True
        )

@pytest.mark.skipif(not HAS_NX, reason='needs networkx')
class TestLeafletFinderGraphMethodLegacy:
    @staticmethod
    def lines2one(lines):
        """Join lines and squash all whitespace"""
        return " ".join(" ".join(lines).split())

    def test_leaflet_finder(self, lipid_heads):
        lfls = LeafletFinder(lipid_heads, pbc=True)
        lfls.run()
        bottom_heads, top_heads = lfls.leaflet_atoms_by_normal
        # Make top be... on top.
        if top_heads.center_of_geometry()[2] < bottom_heads.center_of_geometry()[2]:
            top_heads,bottom_heads = (bottom_heads,top_heads)
        assert_equal(top_heads.indices, np.arange(1,2150,12),
                     err_msg="Found wrong leaflet lipids")
        assert_equal(bottom_heads.indices, np.arange(2521,4670,12),
                     err_msg="Found wrong leaflet lipids")
    
    def test_string_vs_atomgroup_proper(self, universe, lipid_heads):
        lfls_ag = LeafletFinder(lipid_heads, pbc=True)
        lfls_ag.run()
        lfls_string = LeafletFinder(universe, "name PO4", pbc=True)
        lfls_string.run()
        groups_ag = lfls_ag.leaflet_atoms_by_normal
        groups_string = lfls_string.leaflet_atoms_by_normal
        assert_equal(groups_string[0].indices, groups_ag[0].indices)
        assert_equal(groups_string[1].indices, groups_ag[1].indices)
    
    def test_optimize_cutoff(self, lipid_heads):
        cutoff = LeafletFinder.optimize_cutoff(lipid_heads, pbc=True)
        assert_almost_equal(cutoff, 10.5, decimal=4)
    
    def test_pbc_on_off(self, lipid_heads):
        lfls_pbc_on = LeafletFinder(lipid_heads, pbc=True)
        lfls_pbc_on.run()
        lfls_pbc_off = LeafletFinder(lipid_heads, pbc=False)
        lfls_pbc_off.run()
        assert lfls_pbc_on.method.graph.size() > lfls_pbc_off.method.graph.size()
    
    def test_pbc_on_off_difference(self, lipid_heads):
        import networkx

        lfls_pbc_on = LeafletFinder(lipid_heads, cutoff=7, pbc=True)
        lfls_pbc_on.run()
        lfls_pbc_off = LeafletFinder(lipid_heads, cutoff=7, pbc=False)
        lfls_pbc_off.run()
        pbc_on_graph = lfls_pbc_on.method.graph
        pbc_off_graph = lfls_pbc_off.method.graph
        diff_graph = networkx.difference(pbc_on_graph, pbc_off_graph)
        assert_equal(set(diff_graph.edges), {(69, 153), (73, 79),
                                            (206, 317), (313, 319)})
    
    @pytest.mark.parametrize("sparse", [True, False, None])
    def test_sparse_on_off_none(self, lipid_heads, sparse):
        lfls_ag = LeafletFinder(lipid_heads, cutoff=15.0, pbc=True,
                                    sparse=sparse)
        lfls_ag.run()
        assert_almost_equal(len(lfls_ag.method.graph.edges), 1903, decimal=4)
    
    def test_cutoff_update(self, lipid_heads):
        lfls_ag = LeafletFinder(lipid_heads, cutoff=15.0, n_leaflets=10000, pbc=True)
        lfls_ag.run()
        lfls_ag.cutoff = 1.0
        lfls_ag.run()
        assert_almost_equal(lfls_ag.cutoff, 1.0, decimal=4)
        assert_almost_equal(len(lfls_ag.leaflet_atoms_by_normal), 360, decimal=4)
    
    def test_write_selection(self, lipid_heads, tmpdir):
        lfls_ag = LeafletFinder(lipid_heads, cutoff=15.0, pbc=True)
        lfls_ag.run()
        with tmpdir.as_cwd():
            filename = lfls_ag.write_selection('leaflet.vmd')
            expected_output = self.lines2one([
                """# Leaflets found by LeafletFinder(method=graph, select='all', cutoff=15.0 Å, pbc=True)
            # MDAnalysis VMD selection
            atomselect macro leaflet_1 {index 1 13 25 37 49 61 73 85 \\
            97 109 121 133 145 157 169 181 \\
            193 205 217 229 241 253 265 277 \\
            289 301 313 325 337 349 361 373 \\
            385 397 409 421 433 445 457 469 \\
            481 493 505 517 529 541 553 565 \\
            577 589 601 613 625 637 649 661 \\
            673 685 697 709 721 733 745 757 \\
            769 781 793 805 817 829 841 853 \\
            865 877 889 901 913 925 937 949 \\
            961 973 985 997 1009 1021 1033 1045 \\
            1057 1069 1081 1093 1105 1117 1129 1141 \\
            1153 1165 1177 1189 1201 1213 1225 1237 \\
            1249 1261 1273 1285 1297 1309 1321 1333 \\
            1345 1357 1369 1381 1393 1405 1417 1429 \\
            1441 1453 1465 1477 1489 1501 1513 1525 \\
            1537 1549 1561 1573 1585 1597 1609 1621 \\
            1633 1645 1657 1669 1681 1693 1705 1717 \\
            1729 1741 1753 1765 1777 1789 1801 1813 \\
            1825 1837 1849 1861 1873 1885 1897 1909 \\
            1921 1933 1945 1957 1969 1981 1993 2005 \\
            2017 2029 2041 2053 2065 2077 2089 2101 \\
            2113 2125 2137 2149 }
            # MDAnalysis VMD selection
            atomselect macro leaflet_2 {index 2521 2533 2545 2557 2569 2581 2593 2605 \\
            2617 2629 2641 2653 2665 2677 2689 2701 \\
            2713 2725 2737 2749 2761 2773 2785 2797 \\
            2809 2821 2833 2845 2857 2869 2881 2893 \\
            2905 2917 2929 2941 2953 2965 2977 2989 \\
            3001 3013 3025 3037 3049 3061 3073 3085 \\
            3097 3109 3121 3133 3145 3157 3169 3181 \\
            3193 3205 3217 3229 3241 3253 3265 3277 \\
            3289 3301 3313 3325 3337 3349 3361 3373 \\
            3385 3397 3409 3421 3433 3445 3457 3469 \\
            3481 3493 3505 3517 3529 3541 3553 3565 \\
            3577 3589 3601 3613 3625 3637 3649 3661 \\
            3673 3685 3697 3709 3721 3733 3745 3757 \\
            3769 3781 3793 3805 3817 3829 3841 3853 \\
            3865 3877 3889 3901 3913 3925 3937 3949 \\
            3961 3973 3985 3997 4009 4021 4033 4045 \\
            4057 4069 4081 4093 4105 4117 4129 4141 \\
            4153 4165 4177 4189 4201 4213 4225 4237 \\
            4249 4261 4273 4285 4297 4309 4321 4333 \\
            4345 4357 4369 4381 4393 4405 4417 4429 \\
            4441 4453 4465 4477 4489 4501 4513 4525 \\
            4537 4549 4561 4573 4585 4597 4609 4621 \\
            4633 4645 4657 4669 }
    
    """])    
            assert self.lines2one(open('leaflet.vmd').readlines()) == expected_output
    