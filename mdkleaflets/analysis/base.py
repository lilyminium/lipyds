
from typing import Union, Dict, Any
import logging

import numpy as np
from MDAnalysis.core.universe import Universe
from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.analysis.base import AnalysisBase, ProgressBar
from MDAnalysis.analysis.distances import capped_distance

from .leafletfinder import LeafletFinder

logger = logging.getLogger(__name__)

class LeafletAnalysisBase(AnalysisBase):

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
            leafletfinder = LeafletFinder(**leaflet_kwargs)
        self.leafletfinder = leafletfinder

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
        self._first_atoms_inside = sum(ag[0] for ag in sel_in_leafletfinder)
        self._first_atoms_outside = sum(ag[0] for ag in sel_out_leafletfinder)
        
        # placeholder leaflet values
        self.residue_leaflets = np.zeros(self.n_residues, dtype=int)
    

    def _update_leaflets(self):
        """Update the ``residue_leaflets`` attribute for the current frame."""
        self.leafletfinder.run()

        # assign inner residues
        inside_rix = self.residues_inside.resindices
        for rix in inside_rix:
            i = self._resindex_to_analysis_order[rix]
            lf = self.leafletfinder.resindex_to_leaflet[rix]
            self.residue_leaflets[i] = lf

        if not len(self._first_atoms_outside):
            return

        # assign outside residues by neighbors
        box = self.get_box()
        pairs = capped_distance(self._first_atoms_outside.positions,
                                self._first_atoms_inside.positions,
                                max_cutoff=self.leafletfinder.cutoff,
                                box=box, return_distances=False)
        splix = np.where(np.ediff1d(pairs[:, 0]))[0] + 1
        plist = np.split(pairs, splix)
        outside_rix = self.residues_outside.resindices
        for arr in plist:
            i = self._resindex_to_analysis_order[outside_rix[arr[0, 0]]]
            neighbor_rix = inside_rix[arr[:, 1]]
            neighbor_ix = np.array([self._resindex_to_analysis_order[j]
                                    for j in neighbor_rix])
            # get most common neighbors
            leaflet_is = self.residue_leaflets[neighbor_ix]
            most_common = np.bincount(leaflet_is).argmax()
            self.residue_leaflets[i] = most_common

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