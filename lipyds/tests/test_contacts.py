import pytest

from .datafiles import TOY_MIXED_SEGMENTED_BILAYER_RES2

import MDAnalysis as mda
from lipyds import LeafletFinder
from lipyds.analysis.contacts import ContactFraction


class TestContactFraction:

    def test_toy_system(self):
        """
        Test the contact fraction analysis on a toy system.

        This toy system is a mixed-segmented bilayer with 2 residue types.
        In the first frame, the upper layer is totally mixed, i.e.::

            O X O X O X O X O X
            X O X O X O X O X O
            O X O X O X O X O X
            X O X O X O X O X O
            O X O X O X O X O X
            X O X O X O X O X O
            O X O X O X O X O X
            X O X O X O X O X O
            O X O X O X O X O X
            X O X O X O X O X O

        
        The lower layer is totally segmented, i.e. ::

            O O O O O X X X X X
            O O O O O X X X X X
            O O O O O X X X X X
            O O O O O X X X X X
            O O O O O X X X X X
            O O O O O X X X X X
            O O O O O X X X X X
            O O O O O X X X X X
            O O O O O X X X X X
            O O O O O X X X X X

        In the second frame, the layers are reversed.
        """
        u = mda.Universe(TOY_MIXED_SEGMENTED_BILAYER_RES2)
        lf = LeafletFinder(u, select="name PO4", method="graph", cutoff=10)
        cf = ContactFraction(
            u, select="name PO4 GL1 GL2 AM1 AM2 ROH",
            leafletfinder=lf
        )
        cf.run()