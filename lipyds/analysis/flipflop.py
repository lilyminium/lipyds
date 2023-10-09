"""
Lipid Flip Flop
===============

Classes
-------

.. autoclass:: LipidFlipFlop
    :members:

"""

from typing import Union

import numpy as np
from numpy.typing import ArrayLike
from MDAnalysis.core.universe import Universe
from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.analysis.distances import capped_distance, calc_bonds, distance_array
from MDAnalysis.lib.mdamath import norm

from .base import LeafletAnalysisBase
from ..lib.mdautils import get_centers_by_residue
from ..lib.cutils import unwrap_around, mean_unwrap_around


class LipidFlipFlop(LeafletAnalysisBase):
    """Quantify lipid flip-flops between leaflets.

    This method uses an interstitial space between leaflets.
    Transitions between the leaflets to the interstitial space,
    and vice versa, are not counted as translocations. This avoids
    overcounting rapid transitions at the inter-leaflet interface
    that may arise from flaws in the leafletfinder.

    Parameters
    ----------
    universe: AtomGroup or Universe
        :class:`~MDAnalysis.core.universe.Universe` or
        :class:`~MDAnalysis.core.groups.AtomGroup` to operate on.
    select: str (optional)
        A :meth:`Universe.select_atoms` selection string
        for atoms that define the lipid head groups, e.g.
        "name PO4" or "name P*"
    cutoff: float (optional)
        Cutoff distance (ångström) to look for neighbors
    leaflet_width: float (optional)
        Width or z-distance of each leaflet (ångström) for containing
        lipids. The interstitial space is defined as the space
        between the two leaflet widths.
    **kwargs
        Passed to :class:`~lipyds.analysis.base.LeafletAnalysisBase`


    Attributes
    ----------
    flips: numpy.ndarray of ints (n_frames,)
        Number of translocations from outer leaflet to inner leaflet
    flops: numpy.ndarray of ints (n_frames,)
        Number of translocations from inner leaflet to outer leaflet
    translocations: numpy.ndarray of ints (n_frames,)
        Number of total translocations
    flips_by_attr: dictionary of {id: int}
        Dictionary where the key is the label obtained by
        ``group_by_attr`` and the value is the number of translocations
        from outer leaflet to inner leaflet
    flops_by_attr: dictionary of {id: int}
        Dictionary where the key is the label obtained by
        ``group_by_attr`` and the value is the number of translocations
        from inner leaflet to outer leaflet
    translocations_by_attr: dictionary of {id: int}
        Dictionary where the key is the label obtained by
        ``group_by_attr`` and the value is the number of translocations
    """

    upper_i = 0
    lower_i = 1
    inter_i = -1

    def __init__(self, universe: Union[AtomGroup, Universe],
                 select: str = "name ROH",
                 cutoff: float = 25,
                 leaflet_width: float = 8,
                 **kwargs):
        super().__init__(universe, select=select, **kwargs)
        self.leafletfinder.order_leaflets_by_z = True
        self.cutoff = cutoff
        self.leaflet_width = leaflet_width
        self._first_atoms = sum([ag[0] for ag in self.selection.split("residue")])

    def _prepare(self):
        self.results.flipflop_leaflet = np.ones(
            (self.n_frames, self.n_residues), dtype=int
        ) * self.inter_i

    def _get_capped_distances(self, atomgroup: AtomGroup) -> ArrayLike:
        pairs, distances = capped_distance(
            self._first_atoms.positions,
            atomgroup.positions,
            box=self.get_box(),
            max_cutoff=self.cutoff,
            return_distances=True,
        )
        min_dist = np.argsort(distances)
        pairs = pairs[min_dist]
        return pairs

    def _get_unwrapped_coordinates(self, i: int, pairs: ArrayLike,
                                   resindices: ArrayLike,
                                   coordinates: ArrayLike) -> ArrayLike:
        nearest_index = pairs[pairs[:, 0] == i][:, 1][0]
        coords = coordinates[nearest_index]
        center = self._first_atoms[i].position
        # print(coords.shape)
        # print(center)
        # print(resindices[relevant])
        # print(self.get_box()[:3])
        # return mean_unwrap_around(coords, center, resindices[relevant], self.get_box()[:3])
        return coords

    def _single_frame(self):
        row = self.results.flipflop_leaflet[self._frame_index]
        upper_ag = self.leaflet_atomgroups[0]
        lower_ag = self.leaflet_atomgroups[1]
        lower_pairs = self._get_capped_distances(lower_ag)
        upper_pairs = self._get_capped_distances(upper_ag)

        for i in range(self.n_residues):
            if i not in lower_pairs[:, 0]:
                row[i] = self.upper_i
                continue
            elif i not in upper_pairs[:, 0]:
                row[i] = self.lower_i
                continue

            upper_matching_pairs = upper_pairs[upper_pairs[:, 0] == i][:, 1]
            upper_coords = upper_ag.positions[upper_matching_pairs]
            lower_matching_pairs = lower_pairs[lower_pairs[:, 0] == i][:, 1]
            lower_coords = lower_ag.positions[lower_matching_pairs]

            # upper_coords = self._get_unwrapped_coordinates(i, upper_pairs,
            #                                                upper_ag.resindices,
            #                                                upper_ag.positions)
            # lower_coords = self._get_unwrapped_coordinates(i, lower_pairs,
            #                                                lower_ag.resindices,
            #                                                lower_ag.positions)
            central_coord = self._first_atoms[i].position
            central_coord[:2] = 0
            upper_coords[:2] = 0
            lower_coords[:2] = 0

            upper_dist = distance_array(upper_coords, central_coord, box=self.box).mean(axis=0)[-1]
            lower_dist = distance_array(lower_coords, central_coord, box=self.box).mean(axis=0)[-1]

            if upper_dist <= self.leaflet_width:
                row[i] = self.upper_i
            elif lower_dist <= self.leaflet_width:
                row[i] = self.lower_i
            else:
                row[i] = self.inter_i

    def _conclude(self):
        self.results.flips = np.zeros(self.n_residues, dtype=int)
        self.results.flops = np.zeros(self.n_residues, dtype=int)

        if not self.n_frames:
            return

        for i in range(self.n_residues):
            row = self.results.flipflop_leaflet[:, i]
            trans = row[row != -1]
            diff = trans[1:] - trans[:-1]
            self.results.flips[i] = np.sum(diff > 0)  # 0: upper, 1: lower
            self.results.flops[i] = np.sum(diff < 0)

        self.results.translocations = self.results.flips + self.results.flops

        self.results.flips_by_attr = {}
        self.results.flops_by_attr = {}
        self.results.translocations_by_attr = {}

        for each in np.unique(self.ids):
            mask = self.ids == each
            self.results.flips_by_attr[each] = int(sum(self.results.flips[mask]))
            self.results.flops_by_attr[each] = int(sum(self.results.flops[mask]))
            self.results.translocations_by_attr[each] = int(sum(self.results.translocations[mask]))
