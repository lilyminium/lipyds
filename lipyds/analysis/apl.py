"""
Lipid APL
=========

Classes
-------

.. autofunction:: lipid_area

.. autoclass:: AreaPerLipid
    :members:

"""
from typing import Union, Optional, Dict, Any
from MDAnalysis.core.universe import Universe
from MDAnalysis.core.groups import AtomGroup

import numpy as np

from ..leafletfinder import LeafletFinder
from .base import BilayerAnalysisBase, set_results_mean_and_by_attr


class AreaPerLipid(BilayerAnalysisBase):
    """
    Calculate the area of each lipid by projecting it onto a plane with
    neighboring coordinates and creating a Voronoi diagram.

    Parameters
    ----------
    universe: AtomGroup or Universe
        :class:`~MDAnalysis.core.universe.Universe` or
        :class:`~MDAnalysis.core.groups.AtomGroup` to operate on.
    select: str (optional)
        A :meth:`Universe.select_atoms` selection string
        for atoms that define the lipid head groups, e.g.
        "name PO4" or "name P*"
    select_other: str (optional)
        A :meth:`Universe.select_atoms` selection string
        for atoms that should be incorporated in the area calculation
        but that you do not want to calculat areas for.
    cutoff: float (optional)
        Cutoff distance (ångström) to look for neighbors
    cutoff_other: float (optional)
        Cutoff distance (ångström) to look for neighbors in the ``other``
        selection. This is generally shorter than ``cutoff`` -- e.g.
        you may look for only lipid headgroups in ``select``, but all
        protein atoms in ``select_other``.
    **kwargs
        Passed to :class:`~lipyds.analysis.base.BilayerAnalysisBase`
    """

    units = {"Areas": "Å^2"}

    def __init__(self, universe: Union[AtomGroup, Universe],
                 select: Optional[str] = "not protein",
                 select_other: Optional[str] = "protein",
                 leafletfinder: Optional[LeafletFinder] = None,
                 leaflet_kwargs: Dict[str, Any] = {},
                 group_by_attr: str = "resnames",
                 pbc: bool = True, update_leaflet_step: int = 1,
                 normal_axis=[0, 0, 1],
                 cutoff_other: float = 5,
                 **kwargs):
        super().__init__(universe=universe,
                         select=select, select_other=select_other,
                         leafletfinder=leafletfinder,
                         leaflet_kwargs=leaflet_kwargs,
                         group_by_attr=group_by_attr,
                         pbc=pbc, update_leaflet_step=update_leaflet_step,
                         normal_axis=normal_axis,
                         cutoff_other=cutoff_other,
                         augment_bilayer=False,
                         coordinates_from_leafletfinder=False)

    def _prepare(self):
        self.results.areas_by_leaflet = self._nan_array_by_leaflet()
        # self.results.total_areas_by_leaflet = np.full((self.n_leaflets, self.n_frames), np.nan)


    def _single_frame(self):
        frame = self.results.areas_by_leaflet[..., self._frame_index]
        # total = self.results.total_areas_by_leaflet[..., self._frame_index]
        i = 0
        for bilayer in self.bilayers:
            for leaflet in bilayer.leaflets:
                areas = leaflet.compute_all_vertex_areas()
                frame[i][leaflet.analysis_indices] = areas
                i += 1

    @set_results_mean_and_by_attr("areas_by_leaflet")
    def _conclude(self):
        pass
