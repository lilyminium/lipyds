import MDAnalysis as mda
import numpy as np

from lipyds.core.groups import Lipid, LipidGroup

from lipyds.tests.datafiles import (
    SINGLE_POPC_UNWRAPPED,
    SINGLE_POPC_WRAPPED,
    NEURONAL_DDAT,
)

class TestLipid:
    def test_lipid_creation_default(self):
        u = mda.Universe(SINGLE_POPC_WRAPPED)
        headgroup = u.select_atoms("name GL1 GL2")

        lipid = Lipid(headgroup)
        assert lipid.headgroup == headgroup
        assert lipid.headgroup.n_atoms == 2
        assert lipid.tailgroup.n_atoms == 10
        assert lipid.residue == u.residues[0]
        assert lipid.leaflet == -1
        assert lipid.universe is u
    

    def test_unwrapping_positions(self):
        u = mda.Universe(SINGLE_POPC_WRAPPED)
        headgroup = u.select_atoms("name GL1 GL2")
        lipid = Lipid(headgroup)

        wrapped_positions = np.array([
            [  1.3208313 ,   9.08      ,  11.815826  ],
            [174.69952   ,   5.0999985 ,   9.265831  ],
            [  0.27082825,   2.4199982 ,   5.2158356 ],
            [  2.180832  , 174.4887    ,   4.1958313 ],
            [172.12953   ,   2.2799988 ,   3.0258331 ],
            [168.73953   ,   0.9599991 , 170.91324   ],
            [168.78954   , 173.9287    , 168.01324   ],
            [169.11954   , 172.4187    , 164.19324   ],
            [  2.9408417 , 171.7387    ,   0.7158356 ],
            [  3.6408386 , 171.7887    , 166.86322   ],
            [  4.010849  , 170.5787    , 162.92323   ],
            [  5.800827  , 168.3187    , 159.56323   ]
        ])

        assert np.allclose(
            lipid.residue_positions,
            wrapped_positions,
            # lost precision from writing to PDB for test data
            atol=1e-3
        )

        assert lipid._unwrapped_residue_positions_frame == -1

        unwrapped_positions = np.array([
            [  1.3208313 ,   9.08      ,  11.815826  ],
            [ -0.0291748 ,   5.0999985 ,   9.265831  ],
            [  0.27082825,   2.4199982 ,   5.2158356 ],
            [  2.180832  ,  -0.24000072,   4.1958313 ],
            [ -2.5991669 ,   2.2799988 ,   3.0258331 ],
            [ -5.9891663 ,   0.9599991 ,  -0.2041626 ],
            [ -5.939163  ,  -0.8000002 ,  -3.1041641 ],
            [ -5.6091614 ,  -2.3100004 ,  -6.924164  ],
            [  2.9408417 ,  -2.9900017 ,   0.7158356 ],
            [  3.6408386 ,  -2.9400015 ,  -4.2541733 ],
            [  4.010849  ,  -4.1500006 ,  -8.194168  ],
            [  5.800827  ,  -6.410001  , -11.554169  ]
        ])

        assert np.allclose(
            lipid.unwrapped_residue_positions,
            unwrapped_positions,
            atol=1e-3,
        )

        # check frame tracker has changed
        assert lipid._unwrapped_residue_positions_frame == 0

        # check raw positions unaffected
        assert np.allclose(
            lipid.residue_positions,
            wrapped_positions,
            atol=1e-3
        )

        # check headgroup and tailgroup positions
        assert np.allclose(
            lipid.unwrapped_headgroup_positions,
            np.array([
                [  0.27082825,   2.4199982 ,   5.2158356 ],
                [  2.180832  ,  -0.24000072,   4.1958313 ],
            ]),
            atol=1e-3,
        )

        assert np.allclose(
            lipid.unwrapped_tailgroup_positions,
            np.array([
                [  1.3208313 ,   9.08      ,  11.815826  ],
                [ -0.0291748 ,   5.0999985 ,   9.265831  ],
                [ -2.5991669 ,   2.2799988 ,   3.0258331 ],
                [ -5.9891663 ,   0.9599991 ,  -0.2041626 ],
                [ -5.939163  ,  -0.8000002 ,  -3.1041641 ],
                [ -5.6091614 ,  -2.3100004 ,  -6.924164  ],
                [  2.9408417 ,  -2.9900017 ,   0.7158356 ],
                [  3.6408386 ,  -2.9400015 ,  -4.2541733 ],
                [  4.010849  ,  -4.1500006 ,  -8.194168  ],
                [  5.800827  ,  -6.410001  , -11.554169  ]
            ]),
            atol=1e-3,
        )

        # check changing raw positions does not
        # change unwrapped positions but does change raw
        first = lipid._first_headgroup_atom.position
        first[-1] -= 1
        lipid._first_headgroup_atom.position = first

        wrapped_positions2 = np.array([
            [  1.3208313 ,   9.08      ,  11.815826  ],
            [174.69952   ,   5.0999985 ,   9.265831  ],
            [  0.27082825,   2.4199982 ,   4.2158356 ],
            [  2.180832  , 174.4887    ,   4.1958313 ],
            [172.12953   ,   2.2799988 ,   3.0258331 ],
            [168.73953   ,   0.9599991 , 170.91324   ],
            [168.78954   , 173.9287    , 168.01324   ],
            [169.11954   , 172.4187    , 164.19324   ],
            [  2.9408417 , 171.7387    ,   0.7158356 ],
            [  3.6408386 , 171.7887    , 166.86322   ],
            [  4.010849  , 170.5787    , 162.92323   ],
            [  5.800827  , 168.3187    , 159.56323   ]
        ])

        assert np.allclose(
            lipid.residue_positions,
            wrapped_positions2,
            atol=1e-3
        )

        assert np.allclose(
            lipid.unwrapped_residue_positions,
            unwrapped_positions,
            atol=1e-3,
        )

        # check changing the frame does
        # re-trigger unwrapping calculation
        unwrapped_positions2 = np.array([
            [  1.3208313 ,   9.08      ,  11.815826  ],
            [ -0.0291748 ,   5.0999985 ,   9.265831  ],
            [  0.27082825,   2.4199982 ,   4.2158356 ],
            [  2.180832  ,  -0.24000072,   4.1958313 ],
            [ -2.5991669 ,   2.2799988 ,   3.0258331 ],
            [ -5.9891663 ,   0.9599991 ,  -0.2041626 ],
            [ -5.939163  ,  -0.8000002 ,  -3.1041641 ],
            [ -5.6091614 ,  -2.3100004 ,  -6.924164  ],
            [  2.9408417 ,  -2.9900017 ,   0.7158356 ],
            [  3.6408386 ,  -2.9400015 ,  -4.2541733 ],
            [  4.010849  ,  -4.1500006 ,  -8.194168  ],
            [  5.800827  ,  -6.410001  , -11.554169  ]
        ])
        lipid._unwrapped_residue_positions_frame = -1
        assert np.allclose(
            lipid.unwrapped_residue_positions,
            unwrapped_positions2,
            atol=1e-3,
        )


class TestLipidGroup:

    def test_lipidgroup_creation_default(self):
        u = mda.Universe(NEURONAL_DDAT)
        lipids = LipidGroup.from_atom_selections(
            u,
            select_headgroups="name PO4 GL1 GL2 AM1 AM2 ROH"
        )
        assert lipids._unwrapped_residue_positions_frame == -1
        assert len(lipids) == 1230
        assert lipids.n_residues == 1230
        assert lipids.universe is u
        assert lipids.unwrapped_headgroup_centers.shape == (1230, 3)
