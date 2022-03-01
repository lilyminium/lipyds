# import pytest

# import numpy as np
# from numpy.testing import assert_equal, assert_almost_equal
# import MDAnalysis as mda
# from .datafiles import DDAT_POPC_TPR, DDAT_POPC_XTC

# from lipyds import LeafletFinder, LipidFlipFlop


# @pytest.mark.parametrize("stop, translocations", [
#     (1, 0),
#     (10, 132),
#     (50, 716),
# ])
# def test_flipflop(stop, translocations):
#     u = mda.Universe(DDAT_POPC_TPR, DDAT_POPC_XTC)
#     leafletfinder = LeafletFinder(u, select="name PO4")
#     flipflop = LipidFlipFlop(u, select="resname CHOL", leafletfinder=leafletfinder)
#     flipflop.run(stop=stop)
#     assert flipflop.translocations_by_attr["CHOL"] == translocations