"""
Functions for PyVista surfaces.
"""

import numpy as np
from MDAnalysis.analysis import distances as mdadist


def compute_surface_normals(surface, global_normal=[0, 0, 1]):
    surface.compute_normals(point_normals=True, inplace=True)
    average = surface.point_normals.mean(axis=0)
    if np.dot(global_normal, average) < 0:  # obtuse
        surface.flip_normals()


def compute_distances_between_surfaces(target, reference, box=None, vector_length=100):
    distances = np.full(target.n_points, np.nan)
    for i, point in enumerate(target.points): #range(target.n_points):
        point = target.points[i]
        normal = target.point_normals[i]
        vector = vector_length * normal
        origin, end = point - vector, point + vector
        point2, cell = reference.ray_trace(origin, end, first_point=True)
        if not len(point2):
            continue
        distances[i] = mdadist.calc_bonds(point, point2, box=box)
    return distances
