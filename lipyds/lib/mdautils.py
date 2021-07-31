from typing import Optional


import numpy as np
from numpy.typing import ArrayLike
from MDAnalysis.lib.distances import capped_distance
from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.lib.mdamath import norm

from .cutils import unwrap_around, mean_unwrap_around, calc_cosine_similarity, get_centers_by_resindices, get_centers_around_first_by_resindices


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
    if centers is None:
        return get_centers_around_first_by_resindices(selection.positions, selection.resindices, box)
    # splix = np.where(np.ediff1d(selection.resindices))[0]+1
    # sel = np.split(selection.positions, splix)
    # if centers is None:
    #     centers = [x[0] for x in sel]
    # n_points = len(sel)
    centers = np.asarray(centers)
    return get_centers_by_resindices(selection.positions, centers, selection.resindices, box)
    # unwrapped = np.full((n_points, 3), np.nan)
    # for i in range(n_points):
    #     x = sel[i]
    #     # print(x)
    #     if len(x) == 1:
    #         unwrapped[i] = x[0]
    #     else:
    #         unwrapped[i] = mean_unwrap_around(x, centers[i], box)
    # unwrapped = np.array([mean_unwrap_around(x, c, box) for x, c in zip(sel, centers)])
    return unwrapped


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
    pi, pj = tuple(pairs.T)

    # split pairs and distances by residue
    splix = np.where(np.ediff1d(pairs[:, 0]))[0] + 1
    plist = np.split(pairs, splix)
    dlist = np.split(dists, splix)

    # project distances onto orientation vector
    for p, d in zip(plist, dlist):
        i = p[0, 0]
        js = p[1:, 1]  # first is self-to-self
        d = d[1:]
        i_coord = coordinates[i]

        # TODO: can I get around this? slow
        neigh_ = coordinates[js].copy()

        if box is not None:
            unwrap_around(neigh_, i_coord, box)
        neigh_ -= i_coord

        vec = orientations[[i]]
        if np.any(np.isnan(vec)):
            continue

        ang_ = calc_cosine_similarity(vec, neigh_)
        proj = np.abs(d * ang_)
        # weight projected distance by angle_factor
        half = (proj * angle_factor)[0]
        dist_mat[i, js] = half + d

    dist_mat += dist_mat.T  # symmetrize
    dist_mat /= 2
    return dist_mat
