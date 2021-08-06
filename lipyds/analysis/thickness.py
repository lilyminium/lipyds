from scipy import interpolate as spinterp
import numpy as np

from .base import BilayerAnalysisBase
from ..lib import utils


class MembraneThickness(BilayerAnalysisBase):
    r"""Calculate the thickness of one or more bilayers.

    This uses the local normals of each point in the "middle"
    surface. The distance along the local normal to the upper and lower
    leaflets is calculated for the total thickness. If the the local normal
    does not intersect with either leaflet, the value for that point is
    double the other leaflet, or np.nan if the normal does not intersect with
    either leaflet.

    Values are interpolated along a user-specified mesh grid for
    smooth results.

    """

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
        self._setup_axes()

        self.results.thicknesses = []
        self.results.points = []
        for n in range(self.n_bilayers):
            self.results.thicknesses.append([])
            self.results.points.append([])
        shape = (self.n_bilayers, self.n_x, self.n_y, self.n_frames)
        self.results.interpolated_thicknesses = np.full(shape, np.nan)

    def _single_frame(self):
        frame = self._frame_index
        for i, bilayer in enumerate(self.bilayers):
            points, thickness = bilayer.compute_thickness()[:bilayer.middle.n_points]
            mask = ~np.isnan(thickness)
            xy = points[:, self._axes]
            interpolator = self.interpolator(xy[mask], thickness[mask])
            interpolated = interpolator(*self.xy)
            self.results.interpolated_thicknesses[i, ..., frame] = interpolated.T
            self.results.thicknesses[i].append(thickness)
            self.results.points[i].append(points)

    def _conclude(self):
        self.results.thickness_mean = np.full((self.n_bilayers, self.n_frames), np.nan)
        self.results.thickness_std = np.full((self.n_bilayers, self.n_frames), np.nan)
        for i, bilayer in enumerate(self.results.thicknesses):
            self.results.thickness_mean[i] = [np.nanmean(x) for x in bilayer]
            self.results.thickness_std[i] = [np.nanstd(x) for x in bilayer]
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
        self.grid_bounds = operator(cell, axis=0)[self._axes] + self.bin_size

    def _setup_axes(self):
        x, y = self.grid_bounds
        self.x_axis = np.arange(0, x, self.bin_size, dtype=float)
        self.n_x = len(self.x_axis)
        self.y_axis = np.arange(0, y, self.bin_size, dtype=float)
        self.n_y = len(self.y_axis)
        self.grid_bounds = (self.x_axis[-1], self.y_axis[-1])
        self.xy = np.meshgrid(self.x_axis, self.y_axis)
