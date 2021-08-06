import numpy as np
from MDAnalysis.lib import distances as mdadist

from .base import BilayerAnalysisBase, set_results_mean_and_by_attr
from ..lib import mdautils


class LipidTilt(BilayerAnalysisBase):

    units = {"Tilts": "°",
             "Acute Tilts": "°", }

    def __init__(self, universe, *args, select_end=None, **kwargs):
        super().__init__(universe, *args, **kwargs)
        if select_end is None:
            self.ends = self.residues.atoms - self.selection
        else:
            self.ends = self.universe.select_atoms(select_end)

    def _prepare(self):
        self.results.cosine_similarity_by_leaflet = self._nan_array_by_leaflet()

    def _single_frame(self):
        frame = self.results.cosine_similarity_by_leaflet[..., self._frame_index]
        orientations = mdautils.get_orientations(self.selection,
                                                 tailgroups=self.ends,
                                                 box=self.box,
                                                 normalize=True)
        for i, indices in enumerate(self.leaflet_indices):
            middle = self.bilayers[i // 2].middle
            point_indices = self.get_nearest_indices(i)
            normals = middle.surface.point_normals[point_indices]
            cosine = np.einsum("ij,ij->i", orientations[indices], normals)
            frame[i, indices] = cosine


    @set_results_mean_and_by_attr("tilts_by_leaflet",
                                  "acute_tilts_by_leaflet", pre=False)
    def _conclude(self):
        cosine = self.results.cosine_similarity_by_leaflet
        self.results.tilts_by_leaflet = np.rad2deg(np.arccos(cosine))
        self.results.acute_tilts_by_leaflet = np.rad2deg(np.arccos(np.abs(cosine)))
