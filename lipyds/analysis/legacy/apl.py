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
from MDAnalysis.lib.distances import self_capped_distance
from scipy.spatial import ConvexHull, KDTree, Voronoi
from scipy.spatial.transform import Rotation


import numpy as np

from ..leafletfinder import LeafletFinder
from .base import BilayerAnalysisBase, set_results_mean_and_by_attr
from ..lib import mdautils



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
                 cutoff: float = 15,
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
        self.cutoff = cutoff

    def _prepare(self):
        self.results.areas_by_leaflet = self._nan_array_by_leaflet()

    def _single_frame(self):
        frame = self.results.areas_by_leaflet[..., self._frame_index]
        for i, indices in enumerate(self.leaflet_indices):
            bilayer = self.bilayers[i // 2]
            middle = bilayer.middle
            surface = bilayer.leaflets[i % 2]
            point_indices = self.get_nearest_indices(i)
            normals = middle.surface.point_normals[point_indices]
            coordinates = self.leaflet_coordinates[i]
            pairs = self_capped_distance(surface.surface.points,
                                         self.cutoff,
                                        #  box=self.box,
                                         return_distances=False)

            for j in range(surface.n_points):
                ix_ = [j]
                inside = np.where(np.any(pairs == j, axis=1))[0]
                inside = list(np.ravel(pairs[inside]))
                inside += list(surface.get_neighbors([j]))
                ix_ += list(set([x for x in inside if x != j]))

                if len(ix_) < 4:
                    continue

                points = np.array(surface.surface.points[ix_])
                points = points - points[0]

                normal = normals[j]
                z = np.array([0, 0, 1])
                x_ = np.cross(normal, z)
                x_ /= np.linalg.norm(x_)
                y_ = np.cross(normal, x_)
                y_ /= np.linalg.norm(y_)
                current_basis = [x_, y_, normal, points[0]]
                new_basis = [*np.identity(3), points[0]]

                rotation_matrix, rmsd = Rotation.align_vectors(current_basis, new_basis)
                xy = np.matmul(points, rotation_matrix.as_matrix())
                xy -= xy[0]

                vor = Voronoi(xy[:, :2])
                headgroup_cell_int = vor.point_region[0]
                headgroup_cell = vor.regions[headgroup_cell_int]
                # x and y should be ordered clockwise
                x, y = np.array([vor.vertices[x] for x in headgroup_cell]).T
                area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
                area += (x[-1] * y[0] - y[-1] * x[0])
                lipid_area = 0.5 * np.abs(area)

                frame[i, indices[j]] = lipid_area

            


    @set_results_mean_and_by_attr("areas_by_leaflet")
    def _conclude(self):
        pass
