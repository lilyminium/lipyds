import numpy as np
from MDAnalysis.lib import distances as mdadist

from .base import BilayerAnalysisBase, set_results_mean_and_by_attr
from ..lib import mdautils


class LipidTilt(BilayerAnalysisBase):

    def __init__(self, universe, *args, select_end=None, **kwargs):
        super().__init__(universe, *args, **kwargs)
        if select_end is None:
            self.ends = self.residues.atoms - self.selection
        else:
            self.ends = universe.select_atoms(select_end)

    def _prepare(self):
        self.results.cosine_similarity_by_leaflet = self._nan_array_by_leaflet()

    def _single_frame(self):
        frame = self.results.cosine_similarity_by_leaflet[..., self._frame_index]
        orientations = mdautils.get_orientations(self.selection,
                                                 tailgroups=self.ends,
                                                 box=self.box,
                                                 normalize=True)
        # for i, leaflet_ix in enumerate(self.leaflet_indices):
        for i in range(0, self.n_leaflets, 2):
            middle = self.bilayers[i // 2].middle
            ix = middle.analysis_indices
            cosine = np.einsum("ij,ij->i", orientations[ix],
                               middle.point_normals)
            mask_0 = np.where(np.isin(ix, self.leaflet_indices[i]))[0]
            mask_1 = np.where(np.isin(ix, self.leaflet_indices[i + 1]))[0]
            frame[i][mask_0] = cosine[mask_0]
            frame[i + 1][mask_1] = cosine[mask_1]
            # lower = np.where(np.isin(ix, bilayer.lower.analysis_indices))[0]
            # upper = np.where(np.isin(ix, bilayer.upper.analysis_indices))[0]
        # for i, bilayer in enumerate(self.bilayers):
        #     ix = bilayer.middle.analysis_indices
        #     cosine = np.einsum("ij,ij->i", orientations[ix],
        #                        bilayer.middle.point_normals)
        #     lower = np.where(np.isin(ix, bilayer.lower.analysis_indices))[0]
        #     upper = np.where(np.isin(ix, bilayer.upper.analysis_indices))[0]

        #     print(cosine.shape)
        #     print(cosine)
        #     print(lower)
        #     print(bilayer.lower.analysis_indices)

        #     frame[i * 2][lower] = cosine[lower]
        #     frame[i * 2 + 1][upper] = cosine[upper]

    @set_results_mean_and_by_attr("tilts_by_leaflet",
                                  "acute_tilts_by_leaflet", pre=False)
    def _conclude(self):
        cosine = self.results.cosine_similarity_by_leaflet
        self.results.tilts_by_leaflet = np.arccos(cosine)
        self.results.acute_tilts_by_leaflet = np.arccos(np.abs(cosine))