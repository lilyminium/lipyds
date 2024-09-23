import pytest

import numpy as np

from .datafiles import (
    TOY_MIXED_SEGMENTED_BILAYER_RES2,
    MULTI_COMPONENT_BILAYER_FULL,
)

import MDAnalysis as mda
from lipyds import LeafletFinder
from lipyds.analysis.contacts import ContactFraction


class TestContactFraction:

    def test_toy_system_binary(self):
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
        assert cf.results.expected_contact_probability_over_time.shape == (2, 2, 2, 2)

        expected_contact_probability = np.array([
            [0.24747475, 0.50505051],
            [0.50505051, 0.24747475],
        ])

        for leaflet_index, leaflet in enumerate(
            cf.results.expected_contact_probability_over_time
        ):
            for frame_index in range(2): 
                assert np.allclose(
                    leaflet[..., frame_index],
                    expected_contact_probability
                )

        # check counts
        totally_mixed_counts = np.array([
            [81, 180],
            [180, 81]
        ])
        totally_segmented_counts = np.array([
            [157, 28],
            [28, 157]
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

        totally_mixed_probability = np.array([
            [0.23684211, 0.52631579],
            [0.52631579, 0.23684211],
        ])
        totally_segmented_probability = np.array([
            [0.45906433, 0.08187135],
            [0.08187135, 0.45906433],
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
            [0.95703545, 1.04210526],
            [1.04210526, 0.95703545]
        ])
        totally_segmented_contact_fraction = np.array([
            [1.85499463, 0.16210526],
            [0.16210526, 1.85499463]
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
        assert np.allclose(
            cf.results.expected_contact_probability[0],
            expected_contact_probability
        )
        
        assert np.allclose(
            cf.results.expected_contact_probability[1],
            expected_contact_probability
        )
        expected_probability = np.array([
            [0.34795322, 0.30409357],
            [0.30409357, 0.34795322],
        ])
        assert np.allclose(
            cf.results.observed_contact_probability[0], expected_probability
        )
        assert np.allclose(
            cf.results.observed_contact_probability[1], expected_probability
        )
        expected_contact_fraction = np.array([
            [1.40601504, 0.60210526],
            [0.60210526, 1.40601504],
        ])
        assert np.allclose(
            cf.results.contact_fractions[0], expected_contact_fraction
        )
        assert np.allclose(
            cf.results.contact_fractions[1], expected_contact_fraction
        )


    def test_multi_component(self):
        """
        This test system has the following composition:

        # upper leaflet
        - 20 CHL1
        - 0 DDPC
        - 40 DOPA
        - 1 POPC

        # lower leaflet
        - 30 CHL1
        - 25 DDPC
        - 10 DOPA
        - 0 POPC

        # note that upper/lower is reversed when indexing because
        # the lower leaflet has negative coordinates that get wrapped around
        
        """
        u = mda.Universe(MULTI_COMPONENT_BILAYER_FULL)
        selection = "(resname DDPC DOPA POPC and name P O*) or (resname CHL1 and name O*)"
        lf = LeafletFinder(u, select=selection, method="graph", cutoff=20)
        cf = ContactFraction(
            u,
            select=selection,
            leafletfinder=lf,
            cutoff=2.9,
            pbc=True
        )
        cf.run()

        # check group order
        assert list(cf.unique_ids) == ['CHL1', 'DDPC', 'DOPA', 'POPC']


        # check number of lipids in each layer
        assert cf.results.total_counts_over_time.shape == (2, 1)
        assert cf.results.total_counts_over_time[0, 0] == 61
        assert cf.results.total_counts_over_time[1, 0] == 65

        # each layer is 50% POPE 50% POPC
        # check 50 of each
        assert cf.results.group_counts_over_time.shape == (2, 4, 1)
        upper_counts = list(cf.results.group_counts_over_time[0, :, 0])
        lower_counts = list(cf.results.group_counts_over_time[1, :, 0])
        assert upper_counts == [20, 0, 40, 1]
        assert lower_counts == [30, 25, 10, 0]


        # therefore expected probability
        assert cf.results.expected_contact_probability_over_time.shape == (2, 4, 4, 1)

        expected_contact_probability_lower = np.array([
            [0.20913462, 0.36057692, 0.14423077, 0.        ],
            [0.36057692, 0.14423077, 0.12019231, 0.        ],
            [0.14423077, 0.12019231, 0.02163462, 0.        ],
            [0, 0, 0, 0]
        ])
        expected_contact_probability_upper = np.array([
            [0.10382514, 0.        , 0.43715847, 0.01092896],
            [0, 0, 0, 0],
            [0.43715847, 0.        , 0.42622951, 0.02185792],
            [0.01092896, 0.        , 0.02185792, 0.        ]
        ])


        assert np.allclose(
            cf.results.expected_contact_probability_over_time[0, :, :, 0],
            expected_contact_probability_upper
        )
        assert np.allclose(
            cf.results.expected_contact_probability_over_time[1, :, :, 0],
            expected_contact_probability_lower
        )

        # check counts
        expected_counts_upper = np.array([
            [ 386,    0, 3595,   69],
            [   0,    0,    0,    0],
            [3595,    0, 4741,  204],
            [  69,    0,  204,    0],
        ])
        expected_counts_lower = np.array([
            [669, 1644, 1386,    0],
            [1644, 1022, 1598,    0],
            [1386, 1598, 591,    0],
            [   0,    0,    0,    0]
        ])
        
        # upper leaflet, frame 1
        assert np.allclose(
            cf.results.contact_counts_over_time[0, :, :, 0],
            expected_counts_upper
        )
        # lower leaflet, frame 1
        assert np.allclose(
            cf.results.contact_counts_over_time[1, :, :, 0],
            expected_counts_lower
        )

        assert np.isclose(cf.results.total_observed_contacts_over_time[0], 8995)
        assert np.isclose(cf.results.total_observed_contacts_over_time[1], 6910)

        # check observed probability
        expected_probability_lower = np.array([
            [669/6910, 1644/6910, 1386/6910, 0],
            [1644/6910, 1022/6910, 1598/6910, 0],
            [1386/6910, 1598/6910, 591/6910, 0],
            [0, 0, 0, 0]
        ])

        expected_probability_upper = np.array([
            [386/8995, 0, 3595/8995, 69/8995],
            [0, 0, 0, 0],
            [3595/8995, 0, 4741/8995, 204/8995],
            [69/8995, 0, 204/8995, 0]
        ])
        expected_probability_upper_dec = np.array([
            [0.04291273, 0.        , 0.39966648, 0.00767093],
            [0, 0, 0, 0],
            [0.39966648, 0.        , 0.52707059, 0.02267927],
            [0.00767093, 0.        , 0.02267927, 0.        ]
        ])
        expected_probability_lower_dec = np.array([
            [0.09681621, 0.23791606, 0.20057887, 0.        ],
            [0.23791606, 0.14790159, 0.23125904, 0.        ],
            [0.20057887, 0.23125904, 0.08552822, 0.        ],
            [0, 0, 0, 0],
        ])

        assert np.allclose(expected_probability_lower, expected_probability_lower_dec)
        assert np.allclose(expected_probability_upper, expected_probability_upper_dec)
        
        # upper leaflet, frame 1
        assert np.allclose(
            cf.results.observed_contact_probability_over_time[0, :, :, 0],
            expected_probability_upper
        )
        # lower leaflet, frame 1
        assert np.allclose(
            cf.results.observed_contact_probability_over_time[1, :, :, 0],
            expected_probability_lower
        )

        # check contact fractions
        expected_contact_fraction_upper = np.array([
            # 20 CHL1,   0 DDPC,     40 DOPA,     1 POPC
            [0.41331734, np.nan        , 0.91423708, 0.70188994],  # 20 CHL1
            [np.nan, np.nan, np.nan, np.nan], # 0 DDPC
            [0.91423708, np.nan        , 1.2365887 , 1.03757643],  # 40 DOPA
            [0.70188994, np.nan        , 1.03757643, np.nan      ]   # 1 POPC
        ])

        expected_contact_fraction_lower = np.array([
            [0.46293727, 0.65982055, 1.39068017, np.nan ],
            [0.65982055, 1.02545104, 1.92407525, np.nan ],
            [1.39068017, 1.92407525, 3.95330439, np.nan ],
            [np.nan, np.nan, np.nan, np.nan]
        ])
        # upper leaflet, frame 1
        assert np.allclose(
            cf.results.contact_fractions_over_time[0, :, :, 0],
            expected_contact_fraction_upper,
            equal_nan=True
        )
        # lower leaflet, frame 1
        assert np.allclose(
            cf.results.contact_fractions_over_time[1, :, :, 0],
            expected_contact_fraction_lower,
            equal_nan=True
        )

        # check averaging over time
        assert np.allclose(
            cf.results.expected_contact_probability[0],
            expected_contact_probability_upper
        )
        
        assert np.allclose(
            cf.results.expected_contact_probability[1],
            expected_contact_probability_lower
        )
        
        assert np.allclose(
            cf.results.observed_contact_probability[0],
            expected_probability_upper,
        )
        assert np.allclose(
            cf.results.observed_contact_probability[1],
            expected_probability_lower,
        )

        assert np.allclose(
            cf.results.contact_fractions[0],
            expected_contact_fraction_upper,
            equal_nan=True
        )
        assert np.allclose(
            cf.results.contact_fractions[1],
            expected_contact_fraction_lower,
            equal_nan=True
        )