"""
LeafletFinder
=============

Classes
-------

.. autoclass:: LeafletFinder
    :members:
"""

from typing import Optional, Union

import numpy as np
from MDAnalysis.core.universe import Universe
from MDAnalysis.core.groups import AtomGroup

from .grouping import GraphMethod, SpectralClusteringMethod
from .utils import get_centers_by_residue



class LeafletFinder:
    """Identify atoms in the same leaflet of a lipid bilayer.

    You can use a predefined method ("graph", "spectralclustering").
    Alternatively, you can pass in your own function
    as a method.

    Parameters
    ----------
    universe : Universe or AtomGroup
        Atoms to apply the algorithm to
    select : str
        A :meth:`Universe.select_atoms` selection string
        for atoms that define the lipid head groups, e.g.
        universe.atoms.PO4 or "name PO4" or "name P*"
    cutoff : float (optional)
        cutoff distance for computing distances (for the spectral clustering
        method) or determining connectivity in the same leaflet (for the graph
        method). In spectral clustering, it just has to be suitably large to
        cover a significant part of the leaflet, but lower values increase
        computational efficiency. Please see the :func:`optimize_cutoff`
        function for help with values for the graph method. A cutoff is not
        used for the "center_of_geometry" method.
    pbc : bool (optional)
        If ``False``, does not follow the minimum image convention when
        computing distances
    method: str or function (optional)
        method to use to assign groups to leaflets. Choose
        "graph" for :class:`~lipyds.leafletfinder.grouping.GraphMethod`;
        "spectralclustering" for
        :class:`~lipyds.leafletfinder.grouping.SpectralClusteringMethod`;
    **kwargs:
        Passed to ``method``


    Attributes
    ----------
    universe: Universe
    select: str
        Selection string
    selection: AtomGroup
        Atoms that the analysis is applied to
    residues: ResidueGroup
        residues that the analysis is applied to
    headgroups: List of AtomGroup
        Atoms that the analysis is applied to, grouped by residue.
    pbc: bool
        Whether to use PBC or not
    leaflet_indices_by_size: list of list of indices
        List of residue indices in each leaflet. This is the index
        of residues in ``residues``, *not* the canonical ``resindex``
        attribute from MDAnalysis. Leaflets are sorted by size such
        that the largest leaflet is first.
    leaflet_residues_by_size: list of ResidueGroup
        List of ResidueGroups in each leaflet.
        Leaflets are sorted by size such
        that the largest leaflet is first.
    leaflet_atoms_by_size: list of AtomGroup
        List of AtomGroups in each leaflet. 
        Leaflets are sorted by size such
        that the largest leaflet is first.
    leaflet_indices: list of list of indices
        List of residue indices in each leaflet. This is the index
        of residues in ``residues``, *not* the canonical ``resindex``
        attribute from MDAnalysis.
    leaflet_residues: list of ResidueGroup
        List of ResidueGroups in each leaflet. 
        The leaflets are sorted by z-coordinate so that the
        lower-most leaflet is first.
    leaflet_atoms: list of AtomGroup
        List of AtomGroups in each leaflet. 
        The leaflets are sorted by z-coordinate so that the
        lower-most leaflet is first.
    """

    def run(self):
        """
        Actually run the analysis.

        This is its own function to avoid repeating expensive setup in
        __init__ when it's called on every frame.
        """
        clusters = self._method(selection=self.selection,
                                tailgroups=self.tailgroups,
                                cutoff=self.cutoff, pbc=self.pbc,
                                **self.kwargs)
        # output order
        self._output_leaflet_indices = [sorted(x) for x in clusters]
        self._output_leaflet_residues = [self.residues[x] for x in
                                         self._output_leaflet_indices]
        self._output_leaflet_atoms = [x.atoms for x in
                                      self._output_leaflet_residues]

        self.leaflet_indices_by_size = sorted(self._output_leaflet_indices,
                                              key=len, reverse=True)
        self.leaflet_residues_by_size = [self.residues[x] for x in
                                         self.leaflet_indices_by_size]
        self.leaflet_atoms_by_size = [x.atoms for x in
                                      self.leaflet_residues_by_size]
        
        zs = [np.mean(x.positions[:, 2]) for x in
                self.leaflet_atoms_by_size]
        order = np.argsort(zs[:self.n_leaflets])[::-1]

        self.leaflet_indices = [self.leaflet_indices_by_size[x]
                                for x in order]
        self.leaflet_residues = [self.leaflet_residues_by_size[x]
                                 for x in order]
        self.leaflet_atoms = [self.leaflet_atoms_by_size[x]
                              for x in order]

        self.resindex_to_leaflet = {}
        for i, residues in enumerate(self.leaflet_residues):
            for resindex in residues.resindices:
                self.resindex_to_leaflet[resindex] = i

        
    def __init__(self, universe: Union[AtomGroup, Universe],
                 select: Optional[str]='all',
                 select_tailgroups: Optional[str]=None,
                 cutoff: float=40,
                 pbc: bool=True,
                 method: str="spectralclustering",
                 n_leaflets: int=2,
                 **kwargs):
        self.universe = universe.universe
        self.pbc = pbc
        self.n_leaflets = n_leaflets
        self.selection = universe.select_atoms(select, periodic=pbc)
        self.sel_by_residue = self.selection.split("residue")
        self._first_atoms = sum(ag[0] for ag in self.sel_by_residue)
        self.residues = self.selection.residues
        self.n_residues = len(self.residues)
        if select_tailgroups is not None:
            self.tailgroups = self.residues.atoms.select_atoms(select_tailgroups,
                                                               periodic=pbc)
        else:
            self.tailgroups = self.residues.atoms - self.selection

        self.cutoff = cutoff
        self.kwargs = dict(**kwargs)

        if isinstance(method, str):
            method = method.lower().replace('_', '')
        if method == "graph":
            self.method = GraphMethod(self.selection, self.tailgroups,
                                      cutoff=self.cutoff, pbc=self.pbc,
                                      **kwargs)
            self._method = self.method.run
        elif method == "spectralclustering":
            self.method = SpectralClusteringMethod(self.selection, self.tailgroups,
                                                   cutoff=self.cutoff, pbc=self.pbc,
                                                   n_leaflets=self.n_leaflets,
                                                   **kwargs)
            self._method = self.method.run
        else:
            self._method = self.method = method
        self.run()

    def write_selection(self, filename, mode="w", format=None, **kwargs):
        """Write selections for the leaflets to *filename*.

        The format is typically determined by the extension of *filename*
        (e.g. "vmd", "pml", or "ndx" for VMD, PyMol, or Gromacs).

        See :class:`MDAnalysis.selections.base.SelectionWriter` for all
        options.
        """
        sw = selections.get_writer(filename, format)
        with sw(filename, mode=mode,
                preamble=f"Leaflets found by {repr(self)}\n",
                **kwargs) as writer:
            for i, ag in enumerate(self.leaflet_atoms, 1):
                writer.write(ag, name=f"leaflet_{i:d}")

    def __repr__(self):
        return (f"LeafletFinder(method={self.method}, select='{self.select}', "
                f"cutoff={self.cutoff:.1f} Å, pbc={self.pbc})")
