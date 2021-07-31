from typing import Optional


import numpy as np
from numpy.typing import ArrayLike
from MDAnalysis.lib.distances import capped_distance
from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.lib.mdamath import norm

# from .cutils import unwrap_around, mean_unwrap_around, calc_cosine_similarity, get_centers_by_resindices, get_centers_around_first_by_resindices

from .cutils import project_distances, mean_unwrap_around


def get_centers_by_residue(selection: AtomGroup,
                           centers: Optional[ArrayLike] = None,
                           box: Optional[ArrayLike] = None) -> ArrayLike:
    """
    Get center-of-geometry of residues, unwrapping over periodic boundaries

    Parameters
    ----------
    selection: AtomGroup

    """
    if box is None:
        return selection.center(None, compound='residues', pbc=False)
    output = mean_unwrap_around(selection.positions, centers,
                                selection.resindices, box)
    return output


def get_orientations(headgroups: AtomGroup,
                     tailgroups: Optional[AtomGroup] = None,
                     box: Optional[AtomGroup] = None,
                     headgroup_centers: Optional[AtomGroup] = None,
                     normalize: bool = False) -> ArrayLike:
    if headgroup_centers is None:
        headgroup_centers = get_centers_by_residue(headgroups, box=box)
    if tailgroups is None:
        tailgroups = headgroups.residues.atoms - headgroups
    other_centers = get_centers_by_residue(tailgroups, centers=headgroup_centers,
                                           box=box)

    orientations = other_centers - headgroup_centers
    if normalize:
        norms = np.linalg.norm(orientations, axis=1)
        orientations /= norms.reshape(-1, 1)
    return orientations


def get_distances_with_projection(coordinates: ArrayLike,
                                  orientations: ArrayLike,
                                  cutoff: float,
                                  box: Optional[ArrayLike] = None,
                                  angle_factor: float = 1) -> ArrayLike:
    n_coordinates = len(coordinates)
    # set up distance matrix
    filler = (angle_factor + 1) * cutoff
    dist_mat = np.ones((n_coordinates, n_coordinates)) * filler
    dist_mat[np.diag_indices(n_coordinates)] = 0
    pairs, dists = capped_distance(coordinates, coordinates, cutoff, box=box,
                                   return_distances=True)
    a, b = pairs.T
    project_distances(coordinates, orientations, a, b, dists, box, angle_factor)
    dist_mat[a, b] = dists

    dist_mat += dist_mat.T  # symmetrize
    dist_mat /= 2
    return dist_mat
