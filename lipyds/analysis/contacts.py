import typing

import MDAnalysis as mda
from MDAnalysis.lib.distances import self_capped_distance
import numpy as np

from .base import LeafletAnalysisBase


class ContactFraction(LeafletAnalysisBase):
    """
    Calculate the fraction of contacts between groups of residues in a bilayer.

    In a leaflet comprising N lipids with :math:`n_A` molecules of lipid group A,
    the proportion (or ``group_fraction`` of A is:

    .. math::

        Pr(A) = \\frac{n_A}{N}

    The expected probability (or ``expected_contact_probability``) of contacts
    between lipid group A and lipid group B given complete mixing is:

    .. math::

        Pr(A \cap B) = Pr(A) Pr(B)

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
    results.group_counts_over_time : np.ndarray
        The number of residues in each group over time.
        The shape is (n_leaflets, n_groups, n_frames).
    results.total_counts_over_time : np.ndarray
        The total number of residues in each leaflet over time.
        The shape is (n_leaflets, n_frames).
    results.group_fractions_over_time : np.ndarray
        The fraction of residues in each group over time.
        The shape is (n_leaflets, n_groups, n_frames).
    results.expected_contact_probability_over_time : np.ndarray
        The expected probability of contact between each pair of groups
        over time.
        The shape is (n_leaflets, n_groups, n_groups, n_frames).
    results.total_observed_contacts_over_time : np.ndarray
        The total number of observed contacts in each leaflet over time.
        The shape is (n_leaflets, n_frames).
    results.observed_contact_probability_over_time : np.ndarray
        The observed probability of contact between each pair of groups
        over time.
        The shape is (n_leaflets, n_groups, n_groups, n_frames).
    results.contact_fractions_over_time : np.ndarray
        The fraction of observed contacts between each pair of groups
        over time.
        The shape is (n_leaflets, n_groups, n_groups, n_frames).
    results.group_fractions : np.ndarray
        The fraction of residues in each group.
        Unlike ``results.group_fractions_over_time``, this is not per-frame.
        The counts here are *summed* over each frame.
        The shape is (n_leaflets, n_groups).
    results.expected_contact_probability : np.ndarray
        The expected probability of contact between each pair of groups.
        Unlike ``results.expected_contact_probability_over_time``, this is not per-frame.
        The probability here arises from *summed counts* over time.
        The shape is (n_leaflets, n_groups, n_groups).
    results.observed_contact_probability : np.ndarray
        The observed probability of contact between each pair of groups.
        Unlike ``results.observed_contact_probability_over_time``, this is not per-frame.
        The probability here arises from *summed counts* over time.
        The shape is (n_leaflets, n_groups, n_groups).
    results.contact_fractions : np.ndarray
        The fraction of observed contacts between each pair of groups.
        Unlike ``results.contact_fractions_over_time``, this is not per-frame.
        The probability here arises from *summed counts* over time.
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
        self.results.total_counts_over_time = (
            self.results.group_counts_over_time.sum(axis=1)
        )
        self.results.group_fractions_over_time = (
            self.results.group_counts_over_time
            / self.results.total_counts_over_time[:, None]
        )
        self.results.expected_contact_probability_over_time = np.einsum(
            "ijk,ilk->ijlk",
            self.results.group_fractions_over_time,
            self.results.group_fractions_over_time
        )
        self.results.total_observed_contacts_over_time = (
            self.results.contact_counts_over_time.sum(axis=(1, 2)) / 2
        )
        self.results.observed_contact_probability_over_time = (
            self.results.contact_counts_over_time.transpose((1, 2, 0, 3))
            / self.results.total_observed_contacts_over_time
        ).transpose((2, 0, 1, 3))

        self.results.contact_fractions_over_time = (
            self.results.observed_contact_probability_over_time
            / self.results.expected_contact_probability_over_time
        )

        self.results.group_fractions = (
            self.results.group_counts_over_time.sum(axis=-1)
            / self.results.total_counts_over_time.sum(axis=-1)[:, None]
        )
        self.results.expected_contact_probability = np.einsum(
            "ij,il->ijl",
            self.results.group_fractions,
            self.results.group_fractions
        )

        self.results.observed_contact_probability = (
            self.results.contact_counts_over_time.sum(axis=-1)
            / self.results.total_observed_contacts_over_time.sum(
                axis=-1
            )[:, None, None]
        )
        self.results.contact_fractions = (
            self.results.observed_contact_probability
            / self.results.expected_contact_probability
        )

