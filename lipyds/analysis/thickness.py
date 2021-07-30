from scipy import interpolate as spinterp
import numpy as np

from .base import BilayerAnalysisBase
from ..lib import utils


class MembraneThickness(BilayerAnalysisBase):

    def __init__(self, *args, grid_bounds="max", axes=("x", "y"),
                 bin_size=2,
                 interpolator=spinterp.CloughTocher2DInterpolator,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._axes = list(map(utils.axis_to_index, axes))
        self.bin_size = bin_size
        if not isinstance(interpolator, type):
            interpolator = type(interpolator)
        self.interpolator = interpolator
        self.grid_bounds = grid_bounds

    def _prepare(self):
        self._setup_grid()

        self.results.thicknesses = []
        for n in range(self.n_bilayers):
            self.results.thicknesses.append([])
        shape = (self.n_bilayers, self.n_x, self.n_y, self.n_frames)
        self.results.interpolated_thicknesses = np.full(shape, np.nan)

    def _single_frame(self):
        frame = self._frame_index
        for i, bilayer in enumerate(self.bilayers):
            thickness = bilayer.compute_thickness()[:bilayer.middle.n_points]
            mask = ~np.isnan(thickness)
            xy = bilayer.middle.points[:, self._axes]
            interpolator = self.interpolator(xy[mask], thickness[mask])
            print(xy[mask], thickness[mask])
            interpolated = interpolator(*self.xy)
            print(interpolated)
            self.results.interpolated_thicknesses[i, ..., frame] = interpolated
            self.results.thicknesses[i].append(thickness)

    def _conclude(self):
        self.results.thickness_mean = [np.nanmean(x) for x in self.results.thicknesses]
        self.results.thickness_std = [np.nanstd(x) for x in self.results.thicknesses]
        self.results.mean_interpolated = np.nanmean(self.results.interpolated_thicknesses, axis=-1)
        self.results.sd_interpolated = np.nanstd(self.results.interpolated_thicknesses, axis=-1)

    def _setup_grid(self):
        if not isinstance(self.grid_bounds, str):
            return
        if self.grid_bounds == "max":
            operator = np.max
        elif self.grid_bounds == "min":
            operator = np.min
        else:
            operator = np.mean

        cell = [self.universe.dimensions for ts in self.universe.trajectory]
        x, y = operator(cell, axis=0)[self._axes] + self.bin_size
        self.x_axis = np.arange(0, x, self.bin_size, dtype=float)
        self.n_x = len(self.x_axis)
        self.y_axis = np.arange(0, y, self.bin_size, dtype=float)
        self.n_y = len(self.y_axis)
        self.grid_bounds = (self.x_axis[-1], self.y_axis[-1])
        self.xy = np.meshgrid(self.x_axis, self.y_axis)
