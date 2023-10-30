import numpy as np
from MDAnalysis.lib import distances as mdadist

from .base import BilayerAnalysisBase, set_results_mean_and_by_attr
from ..lib import mdautils


class LipidTilt(BilayerAnalysisBase):

    units = {"Tilts": "°",
             "Acute Tilts": "°", }

    def __init__(self, universe, *args, select_end=None, normal="bilayer", cutoff=10, **kwargs):
        super().__init__(universe, *args, **kwargs)
        if select_end is None:
            self.ends = self.residues.atoms - self.selection
        else:
            self.ends = universe.select_atoms(select_end)
        self.cutoff = cutoff

        if not self.selection.n_atoms == self.selection.n_residues:
            raise ValueError("`select` must give a single particle per residue")
        if not self.ends.n_atoms == self.ends.n_residues:
            raise ValueError("`select_end` must give a single particle per residue")

        accepted_values = ("bilayer", "x", "y", "z")
        if normal not in accepted_values:
            raise ValueError(f"normal must be one of {accepted_values}")
        if normal == "x":
            normal = np.array([1, 0, 0])
        elif normal == "y":
            normal = np.array([0, 1, 0])
        elif normal == "z":
            normal = np.array([0, 0, 1])
        self.normal = normal
        

    def _prepare(self):
        self.results.cosine_similarity_by_leaflet = self._nan_array_by_leaflet()

    def _single_frame(self):
        frame = self.results.cosine_similarity_by_leaflet[..., self._frame_index]
        orientations = mdautils.get_orientations(self.selection,
                                                 tailgroups=self.ends,
                                                 box=self.box,
                                                 normalize=True)
        normals = None
        if not isinstance(self.normal, str):
            normals = np.array([self.normal] * len(orientations))
        
        for i, indices in enumerate(self.leaflet_indices):
            if i % 2:
                continue
            if normals is None:
                middle = self.bilayers[i // 2].middle
                middle_points = middle.points

                middle_indices, selection_indices = mdadist.capped_distance(
                    middle_points,
                    self.selection.positions,
                    self.cutoff,
                    box=self.box,
                    return_distances=False,
                ).T
                
                all_normals = []
                for x in range(self.selection.n_atoms):
                    mask = selection_indices == x
                    if not np.any(mask):
                        _, point_indices = middle.kdtree.query([self.selection.positions[x]])
                    else:
                        middle_around = middle_indices[mask]
                        point_indices = np.unique(middle_around)
                    normals = middle.surface.point_normals[point_indices]
                    normal = np.mean(normals, axis=0)
                    all_normals.append(normal)
                
                normals = np.array(all_normals)
            else:
                normals = np.array([self.normal] * len(orientations))

            cosine = np.einsum("ij,ij->i", orientations, normals)
            frame[i, indices] = cosine[indices]
            indices1 = self.leaflet_indices[i + 1]
            frame[i + 1, indices1] = cosine[indices1]


    @set_results_mean_and_by_attr("tilts_by_leaflet",
                                  "acute_tilts_by_leaflet", pre=False)
    def _conclude(self):
        cosine = self.results.cosine_similarity_by_leaflet
        self.results.tilts_by_leaflet = np.rad2deg(np.arccos(cosine))
        self.results.acute_tilts_by_leaflet = np.rad2deg(np.arccos(np.abs(cosine)))
