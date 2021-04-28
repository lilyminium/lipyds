

from collections import defaultdict
import itertools
from typing import Optional, Union

import numpy as np
from MDAnalysis.analysis.distances import capped_distance
from MDAnalysis.core.universe import Universe
from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.lib.c_distances import unwrap_around
from numpy.typing import ArrayLike

from .base import LeafletAnalysisBase
from ..leafletfinder.utils import get_centers_by_residue


def lipid_area(headgroup_coordinate: ArrayLike,
               neighbor_coordinates: ArrayLike,
               other_coordinates: Optional[ArrayLike]=None,
               box: Optional[ArrayLike]=None,
               plot: bool=False) -> float:
    """
    Calculate the area of a lipid by projecting it onto a plane with
    neighboring coordinates and creating a Voronoi diagram.

    Parameters
    ----------
    headgroup_coordinate: numpy.ndarray
        Coordinate array of shape (3,) or (n, 3) of the central lipid
    neighbor_coordinates: numpy.ndarray
        Coordinate array of shape (n, 3) of neighboring lipids to the central lipid.
        These coordinates are used to construct the plane of best fit.
    other_coordinates: numpy.ndarray (optional)
        Coordinate array of shape (n, 3) of neighboring atoms to the central lipid.
        These coordinates are *not* used to construct the plane of best fit, but
        are projected onto it.
    box: numpy.ndarray (optional)
        Box of minimum cell, used for unwrapping coordinates.
    plot: bool (optional)
        Whether to plot the resulting Voronoi diagram.

    Returns
    -------
    area: float
        Area of the central lipid
    
    Raises
    ------
    ValueError
        If a Voronoi cell cannot be constructed for the central lipid, usually
        because too few neighboring lipids have been given.
    """
    from scipy.spatial import Voronoi
    
    # preprocess coordinates
    headgroup_coordinate = np.asarray(headgroup_coordinate)
    if len(headgroup_coordinate.shape) > 1:
        if box is not None:
            headgroup_coordinates = unwrap_around(headgroup_coordinate.copy(),
                                                  headgroup_coordinate[0],
                                                  box)
        headgroup_coordinate = headgroup_coordinates.mean(axis=0)

    if len(neighbor_coordinates) < 2:
        return np.nan

    if box is not None:
        neighbor_coordinates = unwrap_around(neighbor_coordinates.copy(),
                                             headgroup_coordinate,
                                             box)
        if other_coordinates is not None:
            other_coordinates = np.asarray(other_coordinates).copy()
            other_coordinates = unwrap_around(other_coordinates,
                                              headgroup_coordinate,
                                              box)
    points = np.concatenate([[headgroup_coordinate], neighbor_coordinates])
    points -= headgroup_coordinate
    center = points.mean(axis=0)
    points -= center

    Mt_M = np.matmul(points.T, points)
    u, s, vh = np.linalg.linalg.svd(Mt_M)
    # project onto plane
    if other_coordinates is not None:
        points = np.r_[points, other_coordinates-center]
    xy = np.matmul(points, vh[:2].T)
    xy -= xy[0]
    # voronoi
    vor = Voronoi(xy)
    headgroup_cell_int = vor.point_region[0]
    headgroup_cell = vor.regions[headgroup_cell_int]

    if plot:
        from scipy.spatial import voronoi_plot_2d
        import matplotlib.pyplot as plt
        fig = voronoi_plot_2d(vor, show_vertices=False, line_alpha=0.6)
        plt.plot([vor.points[0][0]], [vor.points[0][1]], 'r+', markersize=12)
        plt.show()

    if not all(vertex != -1 for vertex in headgroup_cell):
        raise ValueError("headgroup not bounded by Voronoi cell points: "
                            f"{headgroup_cell}. "
                            "Try including more neighbor points")
    
    # x and y should be ordered clockwise
    x, y = np.array([vor.vertices[x] for x in headgroup_cell]).T
    area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
    area += (x[-1] * y[0] - y[-1] * x[0])
    lipid_area = 0.5 * np.abs(area)

    return lipid_area


class AreaPerLipid(LeafletAnalysisBase):
    """
    Calculate the area of each lipid by projecting it onto a plane with
    neighboring coordinates and creating a Voronoi diagram.

    Parameters
    ----------
    """

    def __init__(self, universe: Union[AtomGroup, Universe], *args,
                 cutoff: float=50, cutoff_other: Optional[float]=None,
                 select_other: Optional[str]=="protein",
                 max_neighbors: int=100, **kwargs):
        super().__init__(universe, *args, **kwargs)
        if select_other is None:
            self.other = self.universe.atoms[[]]
        else:
            self.other = universe.select_atoms(select_other) - self.residues.atoms
        if len(self.other):
            self._get_other_coordinates = self._get_other
        else:
            self._get_other_coordinates = lambda x, y, z: None
        self.cutoff = cutoff
        if cutoff_other is None:
            cutoff_other = cutoff
        self.cutoff_other = cutoff_other
        self.unique_ids = np.unique(self.ids)
        self.max_neighbors = max_neighbors

    def _prepare(self):
        self.areas = np.zeros((self.n_frames, self.n_residues))
        self.areas_by_leaflet = np.zeros((self.n_frames, self.n_leaflets, self.n_residues))
        self.areas_by_leaflet[:] = np.nan
        self.areas_by_attr = []
        for i in range(self.n_leaflets):
            dct = {}
            for each in self.unique_ids:
                dct[each] = []
            self.areas_by_attr.append(dct)

    def _get_other(self, central_xyz: ArrayLike, other: ArrayLike,
                   box: Optional[ArrayLike]=None) -> Optional[ArrayLike]:
        pairs2 = capped_distance(central_xyz, other, self.cutoff_other,
                                 box=box, return_distances=False)
        if len(pairs2):
            other_xyz = other[np.unique(pairs2[:, 1])]
        else:
            other_xyz = None
        
        return other_xyz

    
    def _single_frame(self):
        other = self.other.positions
        box = self.get_box()

        for lf_i in range(self.n_leaflets):
            ag = self.leaflet_atomgroups[lf_i]
            coords = get_centers_by_residue(ag)

            pairs, dists = capped_distance(coords, coords,
                                           self.cutoff,
                                           box=box,
                                           return_distances=True)
            
            if not len(pairs):
                continue
            
            # think this is faster than constructing massive voronoi diagram
            splix = np.where(np.ediff1d(pairs[:, 0]))[0] + 1
            plist = np.split(pairs, splix)
            dlist = np.split(dists, splix)
            d_order = [np.argsort(x) for x in dlist]
            plist = [p[x[:self.max_neighbors+1]] for p, x in zip(plist, d_order)]

            for pairs_ in plist:
                central_i = pairs_[0, 0]
                central_coord = coords[central_i]
                neighbor_coords = coords[pairs_[1:, 1]]

                other_coords = self._get_other_coordinates(central_coord, other, box)

                try:
                    area = lipid_area(central_coord, neighbor_coords,
                                      other_coordinates=other_coords,
                                      box=box)
                except ValueError:
                    area = np.nan

                resindex = ag.residues[central_i].resindex
                residue_index = self._resindex_to_analysis_order[resindex]
                residue_label = self.ids[residue_index]

                self.areas[self._frame_index][residue_index] = area
                self.areas_by_leaflet[self._frame_index][lf_i][residue_index] = area
                self.areas_by_attr[lf_i][residue_label].append(area)
