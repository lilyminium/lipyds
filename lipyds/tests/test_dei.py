import pytest

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
import MDAnalysis as mda
from .datafiles import (NEURONAL_DDAT, NEURONAL_HDAT)

from lipyds import LeafletFinder, LipidEnrichment


@pytest.mark.parametrize("distribution, cutoff, buffer, upper_popc, lower_popc", [
    ("binomial", 6, 0, 0.59, 0.92),
    ("binomial", 6, 2, 0.53, 1.14),
    ("gaussian", 6, 0, 0.59, 0.92),
])
def test_dei(distribution, cutoff, buffer, upper_popc, lower_popc):
    u = mda.Universe(NEURONAL_DDAT)
    leafletfinder = LeafletFinder(u, select="name PO4")
    dei = LipidEnrichment(u, select="name PO4 GL1 GL2 AM1 AM2", leafletfinder=leafletfinder,
                          distribution=distribution, cutoff=cutoff, buffer=buffer,
                          group_by_attr="resnames")
    dei.run(stop=1)
    upper = dei.leaflets_summary[0]
    lower = dei.leaflets_summary[1]
    assert_almost_equal(upper["POPC"]["Mean enrichment"], lower_popc, decimal=2)
    assert_almost_equal(lower["POPC"]["Mean enrichment"], upper_popc, decimal=2)
