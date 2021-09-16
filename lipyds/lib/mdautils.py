from typing import Optional


import numpy as np
from numpy.typing import ArrayLike
from MDAnalysis.lib.distances import capped_distance, distance_array, calc_bonds
from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.lib.mdamath import norm

# from .cutils import unwrap_around, mean_unwrap_around, calc_cosine_similarity, get_centers_by_resindices, get_centers_around_first_by_resindices

from .cutils import project_distances, mean_unwrap_around, unwrap_around


def augment_coordinates(coordinates: ArrayLike,
                        box: ArrayLike, cutoff: float,
                        return_indices: bool = False):
    pbc_pairs, pbc_dists = capped_distance(coordinates, coordinates, cutoff,
                                           return_distances=True, box=box)
    a, b = pbc_pairs.T
    real_dists = calc_bonds(coordinates[a], coordinates[b])
    to_shift = pbc_dists < real_dists
    a, b = pbc_pairs[to_shift].T

    output = np.empty((a.shape[0], 3), dtype=np.float32)
    for i in range(output.shape[0]):
        output[i] = unwrap_around(coordinates[[b[i]]], coordinates[a[i]], box, output[[i]])
    if return_indices:
        return output, b
    return output


def unwrap_coordinates(coordinates: ArrayLike,
                       center: Optional[ArrayLike] = None,
                       box: Optional[ArrayLike] = None) -> ArrayLike:
    coordinates = np.asarray(coordinates, dtype=np.float32).reshape((-1, 3))
    if box is None:
        return coordinates
    if center is None:
        center = coordinates[0]
    output = np.empty((coordinates.shape[0], 3), dtype=np.float32)
    unwrap_around(coordinates, center, box, output)
    return output


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
