import numpy as np
from MDAnalysis.lib._augment import augment_coordinates
from MDAnalysis.lib import distances as mdadist
from MDAnalysis.lib import mdamath
from scipy.spatial import ConvexHull, KDTree
from scipy.spatial.transform import Rotation
import pyvista as pv

from ..lib import pvutils


class Surface:

    def __init__(self, points, other_points=None,
                 cutoff_other=5,
                 box=None, cutoff=30, normal=[0, 0, 1],
                 analysis_indices=None,):
        self.analysis_indices = analysis_indices
        self.box = box

        points = np.asarray(points)
        n_neighbors, n_augmented = 0, 0
        self.n_points = len(points)

        if other_points is not None and len(other_points):
            pairs = mdadist.capped_distance(points, other_points,
                                            cutoff_other,
                                            return_distances=False)
            neighbors = other_points[pairs[:, 1]]
            n_neighbors = len(neighbors)
            points = np.r_[points, neighbors]

        augmented_indices = np.arange(len(points))

        if box is not None:
            points = mdadist.apply_PBC(points, box)
            augmented, indices = augment_coordinates(points, box, cutoff)
            augmented_indices = np.r_[augmented_indices, indices]
            n_augmented = len(augmented)
            points = np.r_[points, augmented]
        else:
            augmented = np.zeros((0, 3), dtype=float)

        self.point_indices = np.arange(self.n_points)

        cloud = pv.PolyData(points)
        surface = cloud.delaunay_2d()
        lipids = np.ones_like(surface.points, dtype=bool)
        lipids[np.arange(n_neighbors) + self.n_points] = False
        not_other = np.where(lipids)[0]
        self.grid = surface.extract_points(not_other)
        self.surface = self.grid.extract_surface()
        self.augmented_indices = augmented_indices[not_other]

        pvutils.compute_surface_normals(self.surface, global_normal=normal)
        self.point_normals = self.surface.point_normals[:self.n_points]
        self.faces = self.surface.faces.reshape((-1, 4))[:, 1:]
        edges = [self.faces[:, :2], self.faces[:, ::2], self.faces[:, 1:]]
        self.edges = np.unique(np.vstack(edges), axis=0)
        self.cell_centers = self.surface.cell_centers()
        self.kdtree = KDTree(self.surface.points)

    @property
    def points(self):
        return self.surface.points[:self.n_points]

    def ray_trace(self, *args, **kwargs):
        return self.surface.ray_trace(*args, **kwargs)

    def compute_distance_to_surface(self, reference, include_outside=False):
        if include_outside:
            obj = self.surface
            reference = reference.surface
        else:
            obj = self
        return pvutils.compute_distances_between_surfaces(obj, reference,
                                                          box=self.box,
                                                          vector_length=120)

    def get_neighbors(self, indices):
        indices = np.ravel(indices)
        matches = np.any(np.isin(self.edges, indices), axis=1)
        neighbors = self.edges[np.where(matches)[0]]
        return np.unique(neighbors)

    def compute_vertex_area(self, index, n_neighbors=2):
        face_indices = np.where(self.faces == index)[0]
        if len(face_indices) < 3:
            # this is not between other points, ignore
            return np.nan
        faces = self.faces[face_indices]

        # get center of triangles -- "voronoi-ish"
        face_centers = self.cell_centers.points[face_indices]
        points = np.r_[[self.surface.points[index]], face_centers]
        points -= points[0]

        normal = self.surface.point_normals[index]

        # rotate and project to xy plane
        z = np.array([0, 0, 1])
        x_ = np.cross(normal, z)
        x_ /= np.linalg.norm(x_)
        y_ = np.cross(normal, x_)
        y_ /= np.linalg.norm(y_)
        current_basis = [x_, y_, normal, points[0]]
        new_basis = [*np.identity(3), points[0]]

        rotation_matrix, rmsd = Rotation.align_vectors(current_basis, new_basis)
        xy = np.matmul(points, rotation_matrix.as_matrix())

        hull = ConvexHull(xy[1:, :2])
        return hull.volume

    def minimum_indices(self, indices):
        return self.augmented_indices[indices]

    def get_nearest_points(self, coordinates,
                           n_nearest_neighbors=1,
                           return_distances=False):
        dist, indices = self.kdtree.query(coordinates, k=n_nearest_neighbors)
        minimum = self.minimum_indices(indices)
        unique, uix = np.unique(minimum, return_index=True)
        if return_distances:
            return unique, dist[uix]
        return unique

    def get_nearest_local_normals(self, coordinates):
        points = self.get_nearest_points(coordinates)
        return self.surface.point_normals[points]

    def compute_all_vertex_areas(self, include_outside=False):
        if include_outside:
            obj = self.surface
        else:
            obj = self
        areas = np.full(self.surface.n_points, np.nan)
        for i in range(obj.n_points):
            areas[i] = self.compute_vertex_area(i)
        self.surface.point_arrays["APL"] = areas
        return areas[:obj.n_points]


class Bilayer:

    def __init__(self, lower_coordinates=None,
                 upper_coordinates=None,
                 other_coordinates=None,
                 upper_indices=None,
                 lower_indices=None,
                 normal=[0, 0, 1], box=None, cutoff=30,
                 cutoff_other=5):
        self.box = box
        self.cutoff = cutoff
        self.cutoff_other = cutoff_other
        self.other_coordinates = other_coordinates
        self.upper_indices = upper_indices
        self.lower_indices = lower_indices
        self.normal = np.asarray(normal)
        self.upper = Surface(upper_coordinates, other_points=other_coordinates,
                             box=box, cutoff=cutoff, normal=self.normal,
                             cutoff_other=cutoff_other,
                             analysis_indices=upper_indices)
        self.lower = Surface(lower_coordinates, other_points=other_coordinates,
                             box=box, cutoff=cutoff, normal=self.normal,
                             cutoff_other=cutoff_other,
                             analysis_indices=lower_indices)
        self.compute_middle()

    def compute_middle(self):
        points = []

        def compute_midpoints(target, reference, operator):
            dist = target.compute_distance_to_surface(reference)
            for pt, normal, d in zip(target.points, target.point_normals, dist):
                if np.dot(normal, [0, 0, 1]) < 0:
                    normal = -normal
                mid_point = operator(pt, normal * (d / 2))
                points.append(mid_point)

        compute_midpoints(self.lower, self.upper, np.add)
        compute_midpoints(self.upper, self.lower, np.subtract)
        points = np.array(points)

        mask = np.all(~np.isnan(points), axis=1)
        if self.upper_indices is not None and self.lower_indices is not None:
            indices = np.concatenate([self.lower_indices, self.upper_indices])
            indices = indices[mask]
        else:
            indices = None

        self.middle = Surface(points[mask],
                              other_points=self.other_coordinates,
                              box=self.box, cutoff=self.cutoff,
                              normal=self.normal,
                              cutoff_other=self.cutoff_other,
                              analysis_indices=indices)

    def compute_thickness(self, include_outside=False):
        upper = self.middle.compute_distance_to_surface(self.upper, include_outside=include_outside)
        lower = self.middle.compute_distance_to_surface(self.lower, include_outside=include_outside)
        thickness = np.nanmean([upper, lower], axis=0) * 2
        padded = np.full(self.middle.surface.n_points, np.nan)
        padded[:len(thickness)] = thickness
        self.middle.surface.point_arrays["Thickness"] = padded
        return thickness
