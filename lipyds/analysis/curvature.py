import pyvista as pv
import numpy as np

from .base import GriddedBilayerAnalysisBase


class Curvature(GriddedBilayerAnalysisBase):

    def _prepare(self):
        self.results.interpolated_surfaces = np.full(self._grid_shape, np.nan)

    def _single_frame(self):
        frame = self._frame_index
        shape = (self.n_x, self.n_y)
        for i, bilayer in enumerate(self.bilayers):
            z = bilayer.middle.interpolator(self._xy)
            self.results.interpolated_surfaces[i, frame] = z.reshape(shape)

    def _conclude(self):
        # framewise
        dx, dy = np.gradient(self.results.interpolated_surfaces, axis=(2, 3))
        d2x, dxy = np.gradient(dx, axis=(2, 3))
        _, d2y = np.gradient(dy, axis=(2, 3))

        dx2 = dx ** 2
        dy2 = dy ** 2
        dsum = 1 + dx2 + dy2
        top = (1 + dx2) * d2y - (2 * dx * dy * dxy) + (1 + dy2) * d2x

        self.results.mean_curvatures = H = top / (dsum ** 1.5)
        self.results.gaussian_curvatures = K = (d2y * d2x - (dxy ** 2)) / (dsum ** 2)
        self.results.maximum_curvatures = H + np.sqrt(H ** 2 - K)
        self.results.minimum_curvatures = H - np.sqrt(H ** 2 - K)

        self.results.average_surfaces = np.nanmean(self.results.interpolated_surfaces,
                                                   axis=1)
        self.results.average_mean = np.nanmean(self.results.mean_curvatures,
                                               axis=1)
        self.results.average_gaussian = np.nanmean(self.results.gaussian_curvatures,
                                                   axis=1)
        self.results.average_min = np.nanmean(self.results.minimum_curvatures,
                                              axis=1)
        self.results.average_max = np.nanmean(self.results.maximum_curvatures,
                                              axis=1)
