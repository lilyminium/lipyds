from typing import Union

import numpy as np
from numpy.typing import ArrayLike
from MDAnalysis.core.universe import Universe
from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.analysis.distances import capped_distance, calc_bonds
from MDAnalysis.lib.mdamath import norm

from .base import LeafletAnalysisBase
from ..leafletfinder.utils import get_centers_by_residue
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
        universe.atoms.PO4 or "name PO4" or "name P*"
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
                 select: str="resname CHOL",
                 cutoff: float=25,
                 leaflet_width: float=8,
                 **kwargs):
        super().__init__(universe, select=select, **kwargs)
        self.leafletfinder.order_leaflets_by_z = True
        self.cutoff = cutoff
        self.leaflet_width = leaflet_width
        self._first_atoms = sum(res.atoms[0] for res in self.residues)
    
    def _prepare(self):
        self.flipflop_leaflet = np.ones((self.n_frames, self.n_residues),
                                        dtype=int) * self.inter_i
    
    def _get_capped_distances(self, atomgroup: AtomGroup) -> ArrayLike:
        return capped_distance(self._first_atoms.positions,
                               atomgroup.positions,
                               box=self.get_box(),
                               max_cutoff=self.cutoff,
                               return_distances=False)

    def _get_unwrapped_coordinates(self, i: int, pairs: ArrayLike,
                                   coordinates: ArrayLike) -> ArrayLike:
        relevant = pairs[pairs[:, 0] == i][:1]
        coords = coordinates[relevant]
        center = self._first_atoms[i].position
        return mean_unwrap_around(coords[0], center, self.get_box()[:3])

    def _single_frame(self):
        row = self.flipflop_leaflet[self._frame_index]
        lower_ag = self.leaflet_atomgroups[0]
        upper_ag = self.leaflet_atomgroups[1]
        lower_pairs = self._get_capped_distances(lower_ag)
        upper_pairs = self._get_capped_distances(upper_ag)

        for i in range(self.n_residues):
            if i not in lower_pairs[:, 0]:
                row[i] = self.upper_i
                continue
            elif i not in upper_pairs[:, 0]:
                row[i] = self.lower_i
                continue
            
            upper_coords = self._get_unwrapped_coordinates(i, upper_pairs,
                                                           upper_ag.positions)
            lower_coords = self._get_unwrapped_coordinates(i, lower_pairs,
                                                           lower_ag.positions)
            central_coord = self._first_atoms[i].position
            central_coord[:2] = 0
            upper_coords[:2] = 0
            lower_coords[:2] = 0
            
            upper_dist = calc_bonds(upper_coords, central_coord)
            lower_dist = calc_bonds(lower_coords, central_coord)
            
            if upper_dist <= self.leaflet_width:
                row[i] = self.upper_i
            elif lower_dist <= self.leaflet_width:
                row[i] = self.lower_i
            else:
                row[i] = self.inter_i

    def _conclude(self):
        self.flips = np.zeros(self.n_residues, dtype=int)
        self.flops = np.zeros(self.n_residues, dtype=int)

        if not self.n_frames:
            return

        for i in range(self.n_residues):
            row = self.flipflop_leaflet[:, i]
            trans = row[row != -1]
            diff = trans[1:] - trans[:-1]
            self.flips[i] = np.sum(diff > 0)  # 0: upper, 1: lower
            self.flops[i] = np.sum(diff < 0)

        self.translocations = self.flips + self.flops

        self.flips_by_attr = {}
        self.flops_by_attr = {}
        self.translocations_by_attr = {}

        for each in np.unique(self.ids):
            mask = self.ids == each
            self.flips_by_attr[each] = int(sum(self.flips[mask]))
            self.flops_by_attr[each] = int(sum(self.flops[mask]))
            self.translocations_by_attr[each] = int(sum(self.translocations[mask]))
