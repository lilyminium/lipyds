import numpy as np
from MDAnalysis.lib import distances as mdadist

from .base import BilayerAnalysisBase, set_results_mean_and_by_attr
from ..lib import mdautils

class ProjectedDisplacement(BilayerAnalysisBase):

    units = {"Displacement": "Å"}

    def _prepare(self):
        self.results.displacement_by_leaflet = self._nan_array_by_leaflet()
    
    def _single_frame(self):
        frame = self.results.displacement_by_leaflet[..., self._frame_index]

        for i, indices in enumerate(self.leaflet_indices):
            middle = self.bilayers[i // 2].middle
            point_indices = self.get_nearest_indices(i)
            normals = middle.surface.point_normals[point_indices]  # already normalized
            midpoints = middle.surface.points[point_indices]

            coordinates = self.leaflet_coordinates[i]
            unwrapped = [mdautils.unwrap_coordinates(x, y, box=self.box)
                         for x, y in zip(coordinates, midpoints)]
            displacement = np.concatenate(unwrapped) - midpoints

            frame[i, indices] = np.einsum('ij,ij->i', displacement, normals)

    @set_results_mean_and_by_attr("displacement_by_leaflet")
    def _conclude(self):
        pass
