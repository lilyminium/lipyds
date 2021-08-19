from typing import Union, Optional, Dict, Any
from MDAnalysis.core.universe import Universe
from MDAnalysis.core.groups import AtomGroup

from scipy import interpolate as spinterp
import numpy as np

from ..leafletfinder import LeafletFinder
from .base import GriddedBilayerAnalysisBase
from ..lib import utils


class MembraneThickness(GriddedBilayerAnalysisBase):
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

    def __init__(self, *args,
                 interpolator=spinterp.CloughTocher2DInterpolator,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(interpolator, type):
            interpolator = type(interpolator)
        self.interpolator = interpolator

    def _prepare(self):
        self.results.thicknesses = []
        self.results.points = []
        for n in range(self.n_bilayers):
            self.results.thicknesses.append([])
            self.results.points.append([])

        self.results.interpolated_thicknesses = np.full(self._grid_shape, np.nan)

    def _single_frame(self):
        frame = self._frame_index
        for i, bilayer in enumerate(self.bilayers):
            points, thickness = bilayer.compute_thickness()[:bilayer.middle.n_points]
            mask = ~np.isnan(thickness)
            xy = points[:, self._axes]
            interpolator = self.interpolator(xy[mask], thickness[mask])
            interpolated = interpolator(*self.xy)
            self.results.interpolated_thicknesses[i, frame] = interpolated.T
            self.results.thicknesses[i].append(thickness)
            self.results.points[i].append(points)

    def _conclude(self):
        self.results.thickness_mean = np.full((self.n_bilayers, self.n_frames), np.nan)
        self.results.thickness_std = np.full((self.n_bilayers, self.n_frames), np.nan)
        for i, bilayer in enumerate(self.results.thicknesses):
            self.results.thickness_mean[i] = [np.nanmean(x) for x in bilayer]
            self.results.thickness_std[i] = [np.nanstd(x) for x in bilayer]
        self.results.mean_interpolated = np.nanmean(self.results.interpolated_thicknesses, axis=1)
        self.results.sd_interpolated = np.nanstd(self.results.interpolated_thicknesses, axis=1)
