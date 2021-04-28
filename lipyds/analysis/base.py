
from typing import Union, Dict, Any, Optional
import logging

import numpy as np
from MDAnalysis.core.universe import Universe
from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.analysis.base import AnalysisBase, ProgressBar
from MDAnalysis.analysis.distances import capped_distance

from ..leafletfinder import LeafletFinder

logger = logging.getLogger(__name__)

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
                 select: Optional[str]="all",
                 leafletfinder: Optional[LeafletFinder]=None,
                 leaflet_kwargs: Dict[str, Any]={},
                 group_by_attr: str="resnames",
                 pbc: bool=True, update_leaflet_step: int=1,
                 **kwargs):
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

        # get selection and labels
        self.selection = universe.select_atoms(select)
        self.sel_by_residue = self.selection.split("residue")
        self.residues = self.selection.residues
        self.n_residues = len(self.residues)
        self.ids = getattr(self.residues, group_by_attr)

        # get mapping dicts
        self._resindex_to_analysis_order = {}
        self._resindex_to_id = {}
        for i, res in enumerate(self.residues):
            self._resindex_to_analysis_order[res.resindex] = i
            self._resindex_to_id[res.resindex] = self.ids[i]
        
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

        # some residues may be selected for analysis, that
        # are not in the leafletfinder.
        sel_in_leafletfinder = []
        sel_out_leafletfinder = []
        for i, sel in enumerate(self.sel_by_residue):
            if self.residues[i] in self.leafletfinder.residues:
                sel_in_leafletfinder.append(sel)
            else:
                sel_out_leafletfinder.append(sel)
        self.sel_inside_leafletfinder = (sum(sel_in_leafletfinder)
                                         or self.selection[[]])
        self.sel_outside_leafletfinder = (sum(sel_out_leafletfinder)
                                          or self.selection[[]])
        self.residues_inside = self.sel_inside_leafletfinder.residues
        self.residues_outside = self.sel_outside_leafletfinder.residues
        self._first_atoms_outside = sum(ag[0] for ag in sel_out_leafletfinder)
        
        # placeholder leaflet values
        self.residue_leaflets = np.zeros(self.n_residues, dtype=int)
    

    def _update_leaflets(self):
        """Update the ``residue_leaflets`` attribute for the current frame."""
        self.leafletfinder.run()
        self.leaflet_atomgroups = {}

        # assign inner residues
        inside_rix = self.residues_inside.resindices
        for rix in inside_rix:
            i = self._resindex_to_analysis_order[rix]
            lf = self.leafletfinder.resindex_to_leaflet[rix]
            self.residue_leaflets[i] = lf

        if not self._first_atoms_outside:
            return

        # assign outside residues by neighbors
        box = self.get_box()
        pairs = capped_distance(self._first_atoms_outside.positions,
                                self.leafletfinder._first_atoms.positions,
                                max_cutoff=self.leafletfinder.cutoff,
                                box=box, return_distances=False)
        splix = np.where(np.ediff1d(pairs[:, 0]))[0] + 1
        plist = np.split(pairs, splix)
        outside_rix = self.residues_outside.resindices
        for arr in plist:
            i = self._resindex_to_analysis_order[outside_rix[arr[0, 0]]]
            neighbor_rix = self.leafletfinder.residues.resindices[arr[:, 1]]
            # neighbor_ix = np.array([self._resindex_to_analysis_order[j]
            #                         for j in neighbor_rix])
            # get most common neighbors
            leaflet_is = [self.leafletfinder.resindex_to_leaflet[r] for r in neighbor_rix]
            most_common = np.bincount(leaflet_is).argmax()
            self.residue_leaflets[i] = most_common
        
        self.leaflet_residues = {i: list() for i in np.unique(self.residue_leaflets)}
        for i, lf_i in enumerate(self.residue_leaflets):
            self.leaflet_residues[lf_i].append(i)
        for lf_i, res_i in self.leaflet_residues.items():
            ag = sum(self.sel_by_residue[i] for i in res_i)
            self.leaflet_atomgroups[lf_i] = ag

    def run(self, start: Optional[int]=None,
            stop: Optional[int]=None, step: Optional[int]=None,
            verbose: Optional[bool]=None):
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
            self._single_frame()
        logger.info("Finishing up")
        self._conclude()
        return self