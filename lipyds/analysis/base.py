
from typing import Union, Dict, Any, Optional, Literal

import logging

import numpy as np
from MDAnalysis.core.universe import Universe
from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.analysis.base import AnalysisBase, ProgressBar
from MDAnalysis.analysis.distances import capped_distance

from lipyds.core.groups import LipidGroup, Leaflet
from lipyds.core.base import ContainsLeafletMixin
from ..leafletfinder import LeafletFinder
from ..lib.mdautils import get_centers_by_residue, unwrap_coordinates
from ..lib.utils import cached_property, get_index_dict, axis_to_index
from .legacy.surface import Bilayer

logger = logging.getLogger(__name__)


class LeafletAnalysisBase(AnalysisBase, ContainsLeafletMixin):
    """
    Base class for leaflet-based analysis.
    
    Parameters
    ----------
    leafletfinder: LeafletFinder
        The leafletfinder to use to assign residues to leaflets.
        This does not have to be the same as the residues
        used in the analysis. If residues are not in the leafletfinder,
        they are assigned to the nearest leaflet by distance.
    universe_or_atomgroup: Universe or AtomGroup
        Atoms or universe to apply the analysis to
    select: str
        Selection string for headgroup atoms to apply the algorithm to.
        Multiple atoms can be selected for each residue.
        For geometry based analyses, usually the center of geometry of headgroups
        will be used as the headgroup position.
    select_tailgroups: str, optional
        Selection string for tailgroup atoms to apply the algorithm to.
        If not given, all atoms that are not in ``select`` are used.
        Tailgroups might not be used in all analysis.
    leaflet_distance_assignment_cutoff: float, optional
        The cutoff for assigning residues to leaflets by distance.
        This is only used if the residues are not in the leafletfinder.
        If a residue is not with ``leaflet_distance_assignment_cutoff`` of
        any leaflet, it is not assigned to a leaflet.
    update_leaflet_step: int, optional
        The number of frames between updating the leaflets.
        Often this can be set higher than 1 to speed up the analysis.
    group_by_attr: str
        The attribute of the residues to group analysis results by.


    Attributes
    ----------
    leafletfinder: LeafletFinder
        The leafletfinder used to assign residues to leaflets.
    lipids: LipidGroup
        The lipids in the analysis.
    residues: mda.core.groups.ResidueGroup
        The residues in the analysis.
    leaflets: list[Leaflet]
        The leaflets of the system.
        This is updated every ``update_leaflet_step`` frames.
    leaflet_distance_assignment_cutoff: float
        The cutoff for assigning residues to leaflets by distance.
        This is only used if the residues are not in the leafletfinder.
        If a residue is not with ``leaflet_distance_assignment_cutoff`` of
        any leaflet, it is not assigned to a leaflet.
    group_by_attr: str
        The attribute of the residues to group analysis results by.
    residue_attributes: np.ndarray
        The ``group_by_attr` attribute of each residue.
        This should be of shape (n_residues,)
    unique_residue_attributes: np.ndarray
        The unique values of ``residue_attributes``.
    residue_attribute_to_index: Dict[Any, int]
        A dictionary mapping each unique value of ``residue_attributes``
        to its index in ``unique_residue_attributes``.
    """

    def __init__(
        self,
        leafletfinder: LeafletFinder,
        universe_or_atomgroup: Union[AtomGroup, Universe],
        select: Optional[str] = "name PO4",
        select_tailgroups: Optional[str] = None,
        leaflet_distance_assignment_cutoff: float = 10,
        update_leaflet_step: int = 1,
        group_by_attr: str = "resnames",
    ):
        super().__init__(universe_or_atomgroup.universe.trajectory)
        self._cache = {}
        self._update_leaflet_step = update_leaflet_step
        self._leafletfinder = leafletfinder
        self._headgroup_atoms = universe_or_atomgroup.select_atoms(select)
        self._universe = self._headgroup_atoms.universe
        self._lipids = LipidGroup.from_atom_selections(
            universe_or_atomgroup,
            select_headgroups=select,
            select_tailgroups=select_tailgroups,
        )

        self.leaflet_distance_assignment_cutoff = leaflet_distance_assignment_cutoff
        # separate lipids that are in leafletfinder and those outside
        leafletfinder_index_to_local_index = {}
        # these must be assigned separately, e.g. by distance

        lipid_indices_outside_leafletfinder = []

        resindex_to_local_index = self._lipids._global_resindex_to_local_index
        resindex_to_leafletfinder_index = self.leafletfinder.lipids._global_resindex_to_local_index

        for lipid_resindex, local_index in resindex_to_local_index.items():
            if lipid_resindex in resindex_to_leafletfinder_index.items():
                leafletfinder_index = resindex_to_leafletfinder_index[lipid_resindex]
                leafletfinder_index_to_local_index[leafletfinder_index] = local_index
            else:
                lipid_indices_outside_leafletfinder.append(local_index)
        
        self._leafletfinder_index_to_local_index = leafletfinder_index_to_local_index
        self._lipid_indices_outside_leafletfinder = np.array(
            lipid_indices_outside_leafletfinder
        )
        self.leaflets = []

        # set up attributes
        self.group_by_attr = group_by_attr
        self.residue_attributes = getattr(self.residues, group_by_attr)
        self.unique_residue_attributes = np.unique(self.residue_attributes)
        self.residue_attribute_to_index = {
            attr: i for i, attr in enumerate(self.unique_residue_attributes)
        }

    def _update_leaflets(self):
        self._cache = {}
        self.leafletfinder.run()
        self.leaflets = [
            Leaflet(self.lipids[indices])
            for indices in self.leaflet_local_indices
        ]
    
    @cached_property
    def leaflet_local_indices(self):
        return self._get_leaflet_indices()


    def _get_leaflet_indices(self) -> list[np.ndarray]:
        leafletfinder_indices = self.leafletfinder.leaflet_local_indices
        local_indices = [
            set([
                self._leafletfinder_index_to_local_index[i]
                for i in indices
            ])
            for indices in leafletfinder_indices
        ]

        # assign external lipids by distance
        for lipid_index in self._lipid_indices_outside_leafletfinder:
            lipid = self.lipids[lipid_index]
            nearest_leaflet_index: int = self.leafletfinder.get_nearest_leaflet_index(
                lipid._first_headgroup_atom.position,
                cutoff=self.leaflet_distance_assignment_cutoff,
            )
            if nearest_leaflet_index != -1:
                local_indices[nearest_leaflet_index].add(lipid_index)
        
        return [
            np.array(sorted(indices))
            for indices in local_indices
        ]

    @property
    def residues(self):
        return self.lipids.residues

    @property
    def n_residues(self):
        return len(self._lipids)
    
    @property
    def lipids(self):
        return self._lipids
    
    @property
    def leafletfinder(self):
        return self._leafletfinder

    def run(self, start: Optional[int] = None,
            stop: Optional[int] = None, step: Optional[int] = None,
            verbose: Optional[bool] = None):
        """Perform the calculation

        Parameters
        ----------
        start : int, optional
            start frame of analysis
        stop : int, optional
            stop frame of analysis
        step : int, optional
            number of frames to skip between each analysed frame
        verbose : bool, optional
            Turn on verbosity
        """
        logger.info("Choosing frames to analyze")
        # if verbose unchanged, use class default
        if verbose is None:
            verbose = getattr(self, '_verbose', False)

        self._setup_frames(self._trajectory, start, stop, step)
        logger.info("Starting preparation")
        self._wrapped_prepare()
        for i, ts in enumerate(ProgressBar(
                self._trajectory[self.start:self.stop:self.step],
                verbose=verbose)):
            self._frame_index = i
            self._ts = ts
            self.frames[i] = ts.frame
            self.times[i] = ts.time
            if not i % self._update_leaflet_step:
                self._update_leaflets()
            self._wrapped_single_frame()
        logger.info("Finishing up")
        self._wrapped_conclude()
        return self

    def _wrapped_single_frame(self):
        self._single_frame()

    def _wrapped_prepare(self):
        self._prepare()

    def _wrapped_conclude(self):
        self._conclude()

    # def results_as_dataframe(self):
    #     import pandas as pd

    #     values = []
    #     leaflets = []
    #     times = []
    #     propnames = []
    #     attrnames = []

    #     for k, v in self.results.items():
    #         if not k.endswith("_by_attr"):
    #             continue

    #         base = k[:-8]
    #         for attrname, array in v.items():
    #             n_leaflets, n_residues, n_frames = array.shape

    #             values_ = np.concatenate(np.concatenate(array))
    #             leaflets_ = np.repeat(np.arange(n_leaflets), n_residues * n_frames)
    #             times_ = np.tile(self.times, n_residues * n_leaflets)

    #             # print(values_.shape, leaflets_.shape, times_.shape, array.shape)
    #             mask = np.where(~np.isnan(values_))[0]
    #             values.append(values_[mask])
    #             leaflets.append(leaflets_[mask])
    #             times.append(times_[mask])
    #             attrnames.extend([attrname] * len(mask))
    #             propnames.extend([base] * len(mask))

    #     values = np.concatenate(values)
    #     leaflets = np.concatenate(leaflets) + 1
    #     times = np.concatenate(times)
    # #     mask = np.where(~np.isnan(values))[0]

    #     dct = {"Leaflet": leaflets,
    #            "Value": values,
    #            "Time": times,
    #            "Label": attrnames,
    #            "Property": propnames}

    #     return pd.DataFrame(dct)

    # def summary_as_dataframe(self):
    #     import pandas as pd

    #     counts = []
    #     means = []
    #     stds = []
    #     variances = []
    #     leaflets = []
    #     propnames = []
    #     attrnames = []
    #     for k, v in self.results.items():
    #         if k.endswith("_by_attr"):
    #             base = [x.capitalize() for x in k[:-8].split("_")]
    #             base = " ".join(base)
    #             for attrname, array in v.items():
    #                 count = (~np.isnan(array)).sum(axis=(1, 2))
    #                 mean = np.nanmean(array, axis=(1, 2))
    #                 std = np.nanstd(array, axis=(1, 2))
    #                 var = np.nanvar(array, axis=(1, 2))
    #                 leaflets_ = np.arange(array.shape[0])
    #                 attrnames_ = [attrname] * len(leaflets_)
    #                 propnames_ = [base] * len(leaflets_)

    #                 leaflets.append(leaflets_)
    #                 counts.append(count)
    #                 means.append(mean)
    #                 stds.append(std)
    #                 variances.append(var)
    #                 attrnames.extend(attrnames_)
    #                 propnames.extend(propnames_)

    #     counts = np.concatenate(counts)
    #     leaflets = np.concatenate(leaflets) + 1
    #     means = np.concatenate(means)
    #     stds = np.concatenate(stds)
    #     variances = np.concatenate(variances)
    #     units = [self.units.get(x, "") for x in propnames]

    #     dct = {"Leaflet": leaflets,
    #            "Mean": means,
    #            "SD": stds,
    #            "Variance": variances,
    #            "Count": counts,
    #            "Label": attrnames,
    #            "Property": propnames,
    #            "Unit": units}

    #     return pd.DataFrame(dct)


class BilayerAnalysisBase(LeafletAnalysisBase):

    def __init__(self, universe: Union[AtomGroup, Universe],
                 select: Optional[str] = "not protein",
                 select_other: Optional[str] = "protein",
                 leafletfinder: Optional[LeafletFinder] = None,
                 leaflet_kwargs: Dict[str, Any] = {},
                 group_by_attr: str = "resnames",
                 pbc: bool = True, update_leaflet_step: int = 1,
                 normal_axis=[0, 0, 1],
                 cutoff_other: float = 5,
                 augment_bilayer: bool = True,
                 augment_max: int = 2000,
                 coordinates_from_leafletfinder: bool = True,
                 augment_cutoff=None,
                 **kwargs):

        super().__init__(universe, select=select,
                         leafletfinder=leafletfinder,
                         leaflet_kwargs=leaflet_kwargs,
                         group_by_attr=group_by_attr,
                         pbc=pbc, update_leaflet_step=update_leaflet_step,
                         normal_axis=normal_axis, **kwargs)
        self.other = universe.select_atoms(select_other)
        self.cutoff_other = cutoff_other
        self.normal_axis = np.asarray(normal_axis)
        self.n_bilayers = self.leafletfinder.n_leaflets // 2
        self.augment_max = augment_max
        self.coordinates_from_leafletfinder = coordinates_from_leafletfinder
        if augment_bilayer and not coordinates_from_leafletfinder:
            self._augment = self._pad_bilayer_coordinates
        else:
            self._augment = lambda x, y: y
        if augment_cutoff is None:
            augment_cutoff = self.leafletfinder.cutoff
        self._augment_cutoff = augment_cutoff

    @property
    def augment_cutoff(self):
        return self._augment_cutoff

    def _pad_bilayer_coordinates(self, index, coordinates):
        if coordinates.shape[0] >= self.augment_max:
            return coordinates
        atoms = self.leafletfinder.leaflet_atoms[index]
        padded = get_centers_by_residue(atoms, box=self.box)
        combined = np.r_[coordinates, padded][:self.augment_max]
        return combined

    def get_nearest_indices(self, leaflet_index):
        middle = self.bilayers[leaflet_index // 2].middle
        leaflet = self.leaflet_coordinates[leaflet_index]
        _, point_indices = middle.kdtree.query(leaflet)
        return point_indices

    def construct_bilayers(self):
        other_coordinates = self.other.positions
        n_leaflets = len(self.leaflet_point_coordinates)
        bilayers = []
        frame = self._frame_index

        if self.coordinates_from_leafletfinder:
            coordinates = self.leafletfinder.leaflet_coordinates
            lower_indices = None
            upper_indices = None
        else:
            coordinates = self.leaflet_coordinates

        for i in range(0, n_leaflets, 2):

            lower = coordinates[i + 1]
            lower = self._augment(i + 1, lower)
            upper = coordinates[i]
            upper = self._augment(i, upper)

            if not self.coordinates_from_leafletfinder:
                lower_indices = self.leaflet_indices[i + 1]
                upper_indices = self.leaflet_indices[i]

            bilayer = Bilayer(lower_coordinates=lower,
                              upper_coordinates=upper,
                              other_coordinates=other_coordinates,
                              lower_indices=lower_indices,
                              upper_indices=upper_indices,
                              box=self.box, cutoff=self.augment_cutoff,
                              normal=self.normal_axis,
                              cutoff_other=self.cutoff_other)
            bilayers.append(bilayer)
        self.bilayers = bilayers

    def _wrapped_single_frame(self):
        self.construct_bilayers()
        self._single_frame()


class GriddedBilayerAnalysisBase(BilayerAnalysisBase):

    def __init__(self, universe: Union[AtomGroup, Universe],
                 select: Optional[str] = "not protein",
                 select_other: Optional[str] = "protein",
                 leafletfinder: Optional[LeafletFinder] = None,
                 leaflet_kwargs: Dict[str, Any] = {},
                 group_by_attr: str = "resnames",
                 pbc: bool = True, update_leaflet_step: int = 1,
                 normal_axis=[0, 0, 1],
                 cutoff_other: float = 5,
                 grid_bounds="max", axes=("x", "y"),
                 bin_size=2,
                 **kwargs):
        super().__init__(universe=universe,
                         select=select, select_other=select_other,
                         leafletfinder=leafletfinder,
                         leaflet_kwargs=leaflet_kwargs,
                         group_by_attr=group_by_attr,
                         pbc=pbc, update_leaflet_step=update_leaflet_step,
                         normal_axis=normal_axis,
                         cutoff_other=cutoff_other,
                         augment_bilayer=False,
                         coordinates_from_leafletfinder=False, **kwargs)

        self._axes = list(map(axis_to_index, axes))
        self.bin_size = bin_size
        self.grid_bounds = grid_bounds

    def _wrapped_prepare(self):
        self._setup_grid()
        self._prepare()

    def _setup_grid(self):
        if isinstance(self.grid_bounds, str):
            if self.grid_bounds == "max":
                operator = np.max
            elif self.grid_bounds == "min":
                operator = np.min
            else:
                operator = np.mean

            cell = [self.universe.dimensions for ts in self.universe.trajectory]
            self.grid_bounds = operator(cell, axis=0)[self._axes] + self.bin_size

        x, y = self.grid_bounds
        self.x_axis = np.arange(0, x, self.bin_size, dtype=float)
        self.n_x = len(self.x_axis)
        self.y_axis = np.arange(0, y, self.bin_size, dtype=float)
        self.n_y = len(self.y_axis)
        self.grid_bounds = (self.x_axis[-1], self.y_axis[-1])
        self._xx, self._yy = np.meshgrid(self.x_axis, self.y_axis)
        self._xy = np.array(list(zip(self._xx.flat, self._yy.flat)))

    @property
    def _grid_shape(self):
        return (self.n_bilayers, self.n_frames, self.n_x, self.n_y)
