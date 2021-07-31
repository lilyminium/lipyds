
from typing import Union, Dict, Any, Optional
from typing_extensions import Literal

import logging

import numpy as np
from MDAnalysis.core.universe import Universe
from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.analysis.base import AnalysisBase, ProgressBar
from MDAnalysis.analysis.distances import capped_distance

from ..leafletfinder import LeafletFinder
from ..lib.mdautils import get_centers_by_residue
from ..lib.utils import cached_property, get_index_dict
from .surface import Bilayer

logger = logging.getLogger(__name__)


def set_results_mean_and_by_attr(*input_names, pre=True):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            if not pre:
                return_value = func(self, *args, **kwargs)
            for name in input_names:
                base = name.split("_by_leaflet", maxsplit=1)[0]
                by_leaflet = getattr(self.results, name)
                values = np.nanmean(by_leaflet, axis=0)
                setattr(self.results, base, values)
                by_attr = self._by_leaflet_to_by_attr(by_leaflet)
                setattr(self.results, f"{base}_by_attr", by_attr)
            if pre:
                return func(self, *args, **kwargs)
            return return_value
        return wrapper
    return decorator


class LeafletAnalysisBase(AnalysisBase):
    """
    Base class for leaflet-based analysis.

    Subclasses should overwrite ``_single_frame()``.

    Parameters
    ----------
    universe: AtomGroup or Universe
        :class:`~MDAnalysis.core.universe.Universe` or
        :class:`~MDAnalysis.core.groups.AtomGroup` to operate on.
    select: str (optional)
        A :meth:`Universe.select_atoms` selection string
        for atoms that define the lipid head groups, e.g.
        "name PO4" or "name P*"
    leafletfinder: LeafletFinder instance (optional)
        A :class:`~lipyds.leafletfinder.leafletfinder.LeafletFinder`
        instance. If this is not provided, a new LeafletFinder
        instance will be created using ``leaflet_kwargs``.
    leaflet_kwargs: dict (optional)
        Arguments to use in creating a new LeafletFinder instance.
        Ignored if an instance is already provided to ``leafletfinder``.
        If ``select`` and ``pbc`` are not present in ``leaflet_kwargs``,
        the values given to ``LeafletAnalysisBase`` are used.
    group_by_attr: str (optional)
        How to group the resulting analysis.
    pbc: bool (optional)
        Whether to use PBC
    update_leaflet_step: int (optional)
        How often to re-run the LeafletFinder. If 1, the LeafletFinder
        is re-run for every frame of the analysis. This can be slow.
        It is unnecessary if you do not have flip-flopping lipids such
        as cholesterol, or you do not care where they are.
    **kwargs:
        Passed to :class:`~MDAnalysis.analysis.base.AnalysisBase`


    Attributes
    ----------
    selection: :class:`~MDAnalysis.core.groups.AtomGroup`
        Selection for the analysis
    sel_by_residue: list of :class:`~MDAnalysis.core.groups.AtomGroup`
        AtomGroups in a list, split up by residue
    residues: :class:`~MDAnalysis.core.groups.ResidueGroup`
        Residues used in the analysis
    n_residues: int
        Number of residues
    ids: numpy.ndarray
        Labels used, obtained from ``group_by_attr``
    leafletfinder: LeafletFinder
    n_leaflets: int
        Number of leaflets
    residue_leaflets: numpy.ndarray of ints, (n_residues,)
        The leaflet index of each residue. Leaflets are sorted by z-coordinate,
        i.e. 0 is the leaflet that has the lowest z-coordinate.
    leaflet_residues: dict of (int, list of ints)
        Dictionary where the key is the leaflet index and the value is a list
        of the residue index in the ``residues`` attribute. This is *not*
        the canonical ``resindex`` attribute in MDAnalysis.
    leaflet_atomgroups: dict of (int, AtomGroup)
        Dictionary where the key is the leaflet index and the value is the
        subset AtomGroup of ``selection`` that is in that leaflet.
    """

    def __init__(self, universe: Union[AtomGroup, Universe],
                 select: Optional[str] = "all",
                 leafletfinder: Optional[LeafletFinder] = None,
                 leaflet_kwargs: Dict[str, Any] = {},
                 group_by_attr: str = "resnames",
                 pbc: bool = True, update_leaflet_step: int = 1,
                 point_coordinates: Literal["average", "closest"] = "average",
                 **kwargs):
        self._cache = {}
        super().__init__(universe.universe.trajectory, **kwargs)
        # store user values
        self.universe = universe.universe
        self.pbc = pbc
        if pbc:
            self.get_box = lambda: self.universe.dimensions
        else:
            self.get_box = lambda: None
        self.group_by_attr = group_by_attr
        self.update_leaflet_step = update_leaflet_step
        self.point_coordinates = point_coordinates

        # get selection and labels
        self.selection = universe.select_atoms(select)
        self.sel_by_residue = self.selection.split("residue")
        self.residues = self.selection.residues
        self.n_residues = len(self.residues)
        self.ids = getattr(self.residues, group_by_attr)
        self.unique_ids = np.unique(self.ids)
        self.id_to_indices = {x: np.where(self.ids == x)[0]
                              for x in self.unique_ids}

        # set up leafletfinder
        if leafletfinder is None:
            leaflet_kwargs = dict(**leaflet_kwargs)  # copy
            if "select" not in leaflet_kwargs:
                leaflet_kwargs["select"] = select
            if "pbc" not in leaflet_kwargs:
                leaflet_kwargs["pbc"] = pbc
            leafletfinder = LeafletFinder(universe, **leaflet_kwargs)
        self.leafletfinder = leafletfinder
        self.n_leaflets = self.leafletfinder.n_leaflets

        # set up outside stuff
        self._outside = self.leafletfinder.get_first_outside_atoms(self.residues)
        _in_dict = get_index_dict(self.leafletfinder.residues.resindices,
                                  self.residues.resindices)
        self._inside_ix = np.array(list(_in_dict.keys()), dtype=int)
        self._inside_lfinder_ix = np.array(list(_in_dict.values()), dtype=int)
        _out_dict = get_index_dict(self._outside.residues.resindices,
                                   self.residues.resindices)
        self._outside_ix = np.array(list(_out_dict.keys()), dtype=int)

        # # placeholder leaflet values
        self.residue_leaflets = np.full(self.n_residues, -1, dtype=int)

    def _update_leaflets(self):
        self._cache = {}
        self.leafletfinder.run()
        self._set_leaflets_with_outside()

    def _set_leaflets_with_outside(self):
        leaflets = np.full(self.n_residues, -1, dtype=int)
        leaflets[self._inside_ix] = self.leafletfinder.residue_leaflets[self._inside_lfinder_ix]
        outside = self.leafletfinder.assign_atoms_by_distance(self._outside)
        leaflets[self._outside_ix] = outside
        self.residue_leaflets = leaflets

    def _set_leaflets_from_finder(self):
        self.residue_leafets = self.leafletfinder.residue_leaflets

    @property
    def box(self):
        return self.get_box()

    @cached_property
    def leaflet_indices(self):
        return [np.where(self.residue_leaflets == i)[0]
                for i in np.unique(self.residue_leaflets)]

    @cached_property
    def leaflet_residues(self):
        return [self.residues[ix] for ix in self.leaflet_indices]

    @cached_property
    def leaflet_atomgroups(self):
        return [sum(self.sel_by_residue[i] for i in ix)
                for ix in self.leaflet_indices]

    @cached_property
    def leaflet_point_coordinates(self):
        if self.point_coordinates == "average":
            return [get_centers_by_residue(ag)
                    for ag in self.leaflet_atomgroups]
        all_coordinates = []
        for i, ix in enumerate(self.leaflet_indices):
            coordinates = []
            leaflet_coordinates = self.leafletfinder.leaflet_atoms[i].positions
            for index in ix:
                atoms = self.sel_by_residue[index]
                if len(atoms) == 1:
                    coordinates.append(atoms.positions[0])
                    continue
                pairs = capped_distance(atoms.positions,
                                        leaflet_coordinates,
                                        self.leafletfinder.cutoff,
                                        return_distances=False)
                atom = atoms[np.bincount(pairs[:, 0])]
                coordinates.append(atom.position)
            all_coordinates.append(np.array(coordinates))
        return all_coordinates

    @property
    def _shape_by_leaflet(self):
        return (self.n_leaflets, self.n_residues, self.n_frames)

    def _nan_array_by_leaflet(self):
        return np.full(self._shape_by_leaflet, np.nan)

    def _by_leaflet_to_by_attr(self, array):
        return {x: array[:, ix, :] for x, ix in self.id_to_indices.items()}

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
        self._prepare()
        for i, ts in enumerate(ProgressBar(
                self._trajectory[self.start:self.stop:self.step],
                verbose=verbose)):
            self._frame_index = i
            self._ts = ts
            self.frames[i] = ts.frame
            self.times[i] = ts.time
            if not i % self.update_leaflet_step:
                self._update_leaflets()
            self._wrapped_single_frame()
        logger.info("Finishing up")
        self._conclude()
        # self._wrapped_conclude()
        return self

    def _wrapped_single_frame(self):
        self._single_frame()

    def results_as_dataframe(self):
        import pandas as pd

        values = []
        leaflets = []
        times = []
        propnames = []
        attrnames = []

        for k, v in self.results.items():
            if not k.endswith("_by_attr"):
                continue

            base = k[:-8]
            for attrname, array in v.items():
                n_leaflets, n_residues, n_frames = array.shape

                values_ = np.concatenate(np.concatenate(array))
                leaflets_ = np.repeat(np.arange(n_leaflets), n_residues * n_frames)
                times_ = np.tile(self.times, n_residues * n_leaflets)

                # print(values_.shape, leaflets_.shape, times_.shape, array.shape)
                mask = np.where(~np.isnan(values_))[0]
                values.append(values_[mask])
                leaflets.append(leaflets_[mask])
                times.append(times_[mask])
                attrnames.extend([attrname] * len(mask))
                propnames.extend([base] * len(mask))

        values = np.concatenate(values)
        leaflets = np.concatenate(leaflets) + 1
        times = np.concatenate(times)
    #     mask = np.where(~np.isnan(values))[0]

        dct = {"Leaflet": leaflets,
               "Value": values,
               "Time": times,
               "Label": attrnames,
               "Property": propnames}

        return pd.DataFrame(dct)

    def summary_as_dataframe(self):
        import pandas as pd

        counts = []
        means = []
        stds = []
        variances = []
        leaflets = []
        propnames = []
        attrnames = []
        for k, v in self.results.items():
            if k.endswith("_by_attr"):
                base = k[:-8]
                for attrname, array in v.items():
                    count = (~np.isnan(array)).sum(axis=(1, 2))
                    mean = np.nanmean(array, axis=(1, 2))
                    std = np.nanstd(array, axis=(1, 2))
                    var = np.nanvar(array, axis=(1, 2))
                    leaflets_ = np.arange(array.shape[0])
                    attrnames_ = [attrname] * len(leaflets_)
                    propnames_ = [base] * len(leaflets_)

                    leaflets.append(leaflets_)
                    counts.append(count)
                    means.append(mean)
                    stds.append(std)
                    variances.append(var)
                    attrnames.extend(attrnames_)
                    propnames.extend(propnames_)

        counts = np.concatenate(counts)
        leaflets = np.concatenate(leaflets) + 1
        means = np.concatenate(means)
        stds = np.concatenate(stds)
        variances = np.concatenate(variances)

        dct = {"Leaflet": leaflets,
               "Mean": means,
               "SD": stds,
               "Variance": variances,
               "Count": counts,
               "Label": attrnames,
               "Property": propnames}

        return pd.DataFrame(dct)


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
                 **kwargs):

        super().__init__(universe, select=select,
                         leafletfinder=leafletfinder,
                         leaflet_kwargs=leaflet_kwargs,
                         group_by_attr=group_by_attr,
                         pbc=pbc, update_leaflet_step=update_leaflet_step,
                         normal_axis=normal_axis)
        self.other = universe.select_atoms(select_other)
        self.cutoff_other = cutoff_other
        self.normal_axis = np.asarray(normal_axis)
        self.n_bilayers = self.leafletfinder.n_leaflets // 2

    def construct_bilayers(self):
        other_coordinates = self.other.positions
        n_leaflets = len(self.leaflet_point_coordinates)
        bilayers = []
        frame = self._frame_index
        for i in range(0, n_leaflets, 2):

            lower = self.leaflet_point_coordinates[i + 1]
            lower_indices = self.leaflet_indices[i + 1]
            upper = self.leaflet_point_coordinates[i]
            upper_indices = self.leaflet_indices[i]
            bilayer = Bilayer(lower_coordinates=lower,
                              upper_coordinates=upper,
                              other_coordinates=other_coordinates,
                              lower_indices=lower_indices,
                              upper_indices=upper_indices,
                              box=self.box, cutoff=self.leafletfinder.cutoff,
                              normal=self.normal_axis,
                              cutoff_other=self.cutoff_other)
            bilayers.append(bilayer)
        self.bilayers = bilayers

    def _wrapped_single_frame(self):
        self.construct_bilayers()
        self._single_frame()
