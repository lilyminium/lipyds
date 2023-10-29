import pytest

import numpy as np

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
            u,
            select="name PO4",
            leafletfinder=lf,
            cutoff=2.9,
            pbc=True
        )
        cf.run()

        # check 100 lipids in each layer
        assert cf.results.total_counts_over_time.shape == (2, 2)
        assert np.allclose(cf.results.total_counts_over_time, 100)

        # each layer is 50% POPE 50% POPC
        # check 50 of each
        assert cf.results.group_counts_over_time.shape == (2, 2, 2)
        assert np.allclose(cf.results.group_counts_over_time, 50)

        # therefore expected probability
        assert cf.results.expected_contact_probability_over_time.shape == (2, 2, 2)
        assert np.allclose(cf.results.expected_contact_probability_over_time, 0.5)

        # check counts
        totally_mixed_counts = np.array([
            [162, 180],
            [180, 162]
        ])
        totally_segmented_counts = np.array([
            [314, 28],
            [28, 314]
        ])
        # upper leaflet, frame 1
        assert np.allclose(
            cf.results.contact_counts_over_time[0, :, :, 0],
            totally_mixed_counts
        )
        # lower leaflet, frame 1
        assert np.allclose(
            cf.results.contact_counts_over_time[1, :, :, 0],
            totally_segmented_counts
        )
        # upper leaflet, frame 2
        assert np.allclose(
            cf.results.contact_counts_over_time[0, :, :, 1],
            totally_segmented_counts
        )
        # lower leaflet, frame 2
        assert np.allclose(
            cf.results.contact_counts_over_time[1, :, :, 1],
            totally_mixed_counts
        )

        assert np.allclose(cf.results.total_observed_contacts_over_time, 342)

        # check observed probability
        totally_mixed_probability = np.array([
            [0.47368421, 0.52631579],
            [0.52631579, 0.47368421],
        ])
        totally_segmented_probability = np.array([
            [0.91812865, 0.08187135],
            [0.08187135, 0.91812865],
        ])

        # upper leaflet, frame 1
        assert np.allclose(
            cf.results.observed_contact_probability_over_time[0, :, :, 0],
            totally_mixed_probability
        )
        # lower leaflet, frame 1
        assert np.allclose(
            cf.results.observed_contact_probability_over_time[1, :, :, 0],
            totally_segmented_probability
        )
        # upper leaflet, frame 2
        assert np.allclose(
            cf.results.observed_contact_probability_over_time[0, :, :, 1],
            totally_segmented_probability
        )
        # lower leaflet, frame 2
        assert np.allclose(
            cf.results.observed_contact_probability_over_time[1, :, :, 1],
            totally_mixed_probability
        )

        # check contact fractions
        totally_mixed_contact_fraction = np.array([
            [0.94736842, 1.05263158],
            [1.05263158, 0.94736842]
        ])
        totally_segmented_contact_fraction = np.array([
            [1.83625731, 0.16374269],
            [0.16374269, 1.83625731]
        ])
        # upper leaflet, frame 1
        assert np.allclose(
            cf.results.contact_fractions_over_time[0, :, :, 0],
            totally_mixed_contact_fraction
        )
        # lower leaflet, frame 1
        assert np.allclose(
            cf.results.contact_fractions_over_time[1, :, :, 0],
            totally_segmented_contact_fraction
        )
        # upper leaflet, frame 2
        assert np.allclose(
            cf.results.contact_fractions_over_time[0, :, :, 1],
            totally_segmented_contact_fraction
        )
        # lower leaflet, frame 2
        assert np.allclose(
            cf.results.contact_fractions_over_time[1, :, :, 1],
            totally_mixed_contact_fraction
        )

        # check averaging over time
        assert np.allclose(cf.results.expected_contact_probability, 0.5)
        expected_probability = np.array([
            [0.69590643, 0.30409357],
            [0.30409357, 0.69590643],
        ])
        assert np.allclose(
            cf.results.observed_contact_probability[0], expected_probability
        )
        assert np.allclose(
            cf.results.observed_contact_probability[1], expected_probability
        )
        expected_contact_fraction = np.array([
            [1.39181287, 0.60818713],
            [0.60818713, 1.39181287],
        ])
        assert np.allclose(
            cf.results.contact_fractions[0], expected_contact_fraction
        )
        assert np.allclose(
            cf.results.contact_fractions[1], expected_contact_fraction
        )