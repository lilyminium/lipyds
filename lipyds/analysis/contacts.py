import itertools
import typing

import MDAnalysis as mda
from MDAnalysis.lib.distances import self_capped_distance
import numpy as np

from .base import LeafletAnalysisBase


class ContactFraction(LeafletAnalysisBase):
    """
    Calculate the fraction of contacts between groups of residues in a bilayer.

    In a leaflet comprising N lipids with :math:`n_A` molecules of lipid group A,
    and :math:`n_B` molecules of lipid group B, the `expected_contact_probability`
    between a single molecule `b` and any molecule in group `A` is:

    .. math::

        Pr(A, B) = \\frac{n_a \\times n_b}{N(N - 1)}

    The `expected_contact_probability` between a single molecule `b` and any
    other molecule in group `B` is:

    .. math::

        Pr(B) = \\frac{n_b \\times (n_b - 1)}{N(N - 1)}

    That is to say, if only a single molecule of group `B` is present, the
    probability of contact between `b` and any other molecule in group `B` is 0.

    The observed fraction (or ``observed_contact_probability``) of contacts
    between lipid group A and lipid group B is the ratio of observed contacts
    (``contact_counts``) to the total number of contacts (``total_contacts``).

    The contact fraction (CF) ratio is then calculated as the ratio of
    observed fraction (or ``observed_contact_probability``) between A and B
    to the expected probability of contacts (or ``expected_contact_probability``).
    That is, where :math:`C_{AB}` represents the number of contacts
    between lipids in group A and lipids in group B (``contact_counts``), and where
    C represents the total number of contacts (``total_contacts``) between lipids:

    .. math::

        CF_{AB} = \\frac{C_{AB}}{C} \\frac{1}{Pr(A \cap B)}

    The contact fraction is bounded between [0, infinity).
    A value of nan indicates that the expected contact probability is 0.
    A value of 1 indicates that the observed contact probability is equal
    to the expected contact probability. A value < 1 indicates that the observed
    contact probability is less than the expected contact probability.
    A value > 1 indicates that the observed contact probability is greater
    than the expected contact probability.


    Parameters
    ----------
    universe : mda.Universe or mda.AtomGroup
        The universe to analyze.
    cutoff : float
        The cutoff distance for determining contacts.
    group_by_attr : str
        The attribute of the residues to group by. Default is 'resname'.


    Attributes
    ----------
    results : Namespace
        The results of the analysis.
    results.contact_counts_over_time : np.ndarray
        The number of contacts between each pair of groups over time.
        The shape is (n_leaflets, n_groups, n_groups, n_frames).
        results.contact_counts_over_time[i, j, k, l] is the number of contacts
        between group j and group k in leaflet i at frame l.
        Note that these means that summing the counts per leaflet and frame
        will *double-count* off-diagonal contacts. However, the diagonal
        is not double-counted.
    results.group_counts_over_time : np.ndarray
        The number of residues in each group over time.
        The shape is (n_leaflets, n_groups, n_frames).
        results.group_counts_over_time[i, j, k] is the number of residues
        in group j in leaflet i at frame k.
    results.total_counts_over_time : np.ndarray
        The total number of residues in each leaflet over time.
        The shape is (n_leaflets, n_frames).
        results.total_counts_over_time[i, j] is the total number of residues
        in leaflet i at frame j.
    results.expected_contact_probability_over_time : np.ndarray
        The expected proportion of contacts for each group.
        The shape is (n_leaflets, n_groups, n_frames).
        results.expected_contact_probability_over_time[i, j, k] is the expected
        proportion of contacts between group j and group k in leaflet i at frame k.
        Note that this means that summing the probabilities per leaflet and frame
        will *double-count* off-diagonal contacts. However, the diagonal
        is not double-counted. Summing the upper or lower triangle will give a total of 1.
    results.total_observed_contacts_over_time : np.ndarray
        The total number of observed contacts in each leaflet over time.
        The shape is (n_leaflets, n_frames).
        results.total_observed_contacts_over_time[i, j] is the total number of
        observed contacts in leaflet i at frame j.
    results.observed_contact_probability_over_time : np.ndarray
        The observed probability of contact between each pair of groups
        over time.
        The shape is (n_leaflets, n_groups, n_groups, n_frames).
        results.observed_contact_probability_over_time[i, j, k, l] is the observed
        probability of contact between group j and group k in leaflet i at frame l.
        Note that this means that summing the probabilities per leaflet and frame
        will *double-count* off-diagonal contacts. However, the diagonal
        is not double-counted. Summing the upper or lower triangle will give a total of 1.
    results.contact_fractions_over_time : np.ndarray
        The contact fraction between each pair of groups over time.
        The shape is (n_leaflets, n_groups, n_groups, n_frames).
        results.contact_fractions_over_time[i, j, k, l] is the contact fraction 
        between group j and group k in leaflet i at frame l. 
    results.expected_contact_probability : np.ndarray
        The expected probability of contact between each pair of groups.
        Unlike ``results.expected_contact_probability_over_time``, this is not per-frame.
        The probability here arises from *summed counts* over time, not the mean.
        The shape is (n_leaflets, n_groups, n_groups).
        results.expected_contact_probability[i, j, k] is the expected probability
        of contact between group j and group k in leaflet i.
    results.observed_contact_probability : np.ndarray
        The observed probability of contact between each pair of groups.
        Unlike ``results.observed_contact_probability_over_time``, this is not per-frame.
        The probability here arises from *summed counts* over time, not the mean.
        The shape is (n_leaflets, n_groups, n_groups).
        results.observed_contact_probability[i, j, k] is the observed probability
        of contact between group j and group k in leaflet i.
    results.contact_fractions : np.ndarray
        The fraction of observed contacts between each pair of groups.
        Unlike ``results.contact_fractions_over_time``, this is not per-frame.
        The probability here arises from *summed counts* over time, not the mean.
        The shape is (n_leaflets, n_groups, n_groups).
    """
    def __init__(
        self,
        universe: typing.Union[
            mda.core.groups.AtomGroup,
            mda.core.universe.Universe
        ],
        cutoff: float = 12,
        **kwargs
    ):
        super().__init__(universe, **kwargs)

        self.cutoff = cutoff

    def _prepare(self):
        n_ids = len(self.unique_ids)
        self.results.contact_counts_over_time = np.zeros(
            (self.n_leaflets, n_ids, n_ids, self.n_frames),
            dtype=int
        )
        self.results.group_counts_over_time = np.zeros(
            (self.n_leaflets, n_ids, self.n_frames),
            dtype=int
        )
    
    def _single_frame(self):
        for leaflet_index, residues in enumerate(self.leaflet_residues):
            atomgroup = residues.atoms
            residues_ids = getattr(residues, self.group_by_attr)
            for rid, id_index in self._unique_id_to_index.items():
                count = np.sum(residues_ids == rid)
                self.results.group_counts_over_time[
                    leaflet_index,
                    id_index,
                    self._frame_index
                ] += count

            # get all contacts
            left_atom_indices, right_atom_indices = self_capped_distance(
                atomgroup.positions,
                self.cutoff,
                box=self.box,
                return_distances=False,
            ).T
            left_resindices = atomgroup[left_atom_indices].resindices
            right_resindices = atomgroup[right_atom_indices].resindices

            global_to_local_resindex = {
                resindex: i for i, resindex in enumerate(residues.resindices)
            }
            local_resindex_to_id_index = {
                i: self._unique_id_to_index[rid]
                for i, rid in enumerate(residues_ids)
            }

            for left, right in zip(left_resindices, right_resindices):
                if left == right:
                    continue
                left_local = global_to_local_resindex[left]
                right_local = global_to_local_resindex[right]
                left_id_index = local_resindex_to_id_index[left_local]
                right_id_index = local_resindex_to_id_index[right_local]

                self.results.contact_counts_over_time[
                    leaflet_index,
                    left_id_index,
                    right_id_index,
                    self._frame_index
                ] += 1
                self.results.contact_counts_over_time[
                    leaflet_index,
                    right_id_index,
                    left_id_index,
                    self._frame_index
                ] += 1

    def _conclude(self):
        n_groups = len(self.unique_ids)
        # (n_groups, n_groups, n_leaflets, n_frames)
        original_contact_counts_over_time = np.array(self.results.contact_counts_over_time)
        # don't doublecount diagonal in user-facing results
        contact_counts_over_time = self.results.contact_counts_over_time.transpose(
            1, 2, 0, 3
        )
        for i in range(n_groups):
            contact_counts_over_time[i, i] //= 2
        # (n_leaflets, n_groups, n_groups, n_frames)
        self.results.contact_counts_over_time = contact_counts_over_time.transpose(
            2, 0, 1, 3
        )
        
        # (n_leaflets, n_frames)
        self.results.total_counts_over_time = (
            self.results.group_counts_over_time.sum(axis=1)
        )

        # temporarily transpose as this is confusing
        # (n_groups, n_leaflets, n_frames)
        full_counts = self.results.group_counts_over_time.transpose(1, 0, 2)

        # (n_leaflets, n_frames)
        total_counts = self.results.total_counts_over_time
        lower = total_counts * (total_counts - 1)

        # (n_groups, n_groups, n_leaflets, n_frames)
        outer_product = np.einsum(
            'ijk,ljk->iljk', full_counts, full_counts
        )
        # subtract off the diagonal
        diag_indices = np.diag_indices(n_groups)
        outer_product[diag_indices] -= full_counts
        # double the off-diagonals
        outer_product += outer_product.transpose(1, 0, 2, 3)
        outer_product[diag_indices] //= 2

        self.results.expected_contact_probability_over_time = (
            outer_product / lower
        ).transpose(2, 0, 1, 3)

        # (n_leaflets, n_frames)
        total_observed_contacts_over_time = original_contact_counts_over_time.sum(axis=(1, 2))
        # again don't double count in user-facing results
        self.results.total_observed_contacts_over_time = total_observed_contacts_over_time / 2
        # self.results.observed_contact_probability_over_time = (
        #     original_contact_counts_over_time.transpose((1, 2, 0, 3))
        #     / total_observed_contacts_over_time
        # ).transpose((2, 0, 1, 3))

        self.results.observed_contact_probability_over_time = (
            contact_counts_over_time / self.results.total_observed_contacts_over_time

        ).transpose((2, 0, 1, 3))

        self.results.contact_fractions_over_time = (
            self.results.observed_contact_probability_over_time
            / self.results.expected_contact_probability_over_time
        )

        # expected contact ability without averaging over frames
        # just sum over time
        # (n_groups, n_groups, n_leaflets)
        contact_counts = outer_product.sum(axis=-1)
        lower_2 = lower.sum(axis=-1)
        self.results.expected_contact_probability = (
            contact_counts / lower_2
        ).transpose(2, 0, 1)

        contact_counts = self.results.contact_counts_over_time.sum(axis=-1)
        total_observed_contacts = self.results.total_observed_contacts_over_time.sum(axis=-1)

        self.results.observed_contact_probability = (
            contact_counts / total_observed_contacts[..., None, None]
        )
        
        self.results.contact_fractions = (
            self.results.observed_contact_probability
            / self.results.expected_contact_probability
        )

