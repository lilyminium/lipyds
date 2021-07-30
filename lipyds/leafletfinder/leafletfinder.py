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
from MDAnalysis.selections import get_writer
from MDAnalysis.analysis.distances import capped_distance

from .grouping import GraphMethod, SpectralClusteringMethod
from ..lib.utils import cached_property


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

    def __init__(self, universe: Union[AtomGroup, Universe],
                 select: Optional[str] = 'all',
                 select_tailgroups: Optional[str] = None,
                 cutoff: float = 40,
                 pbc: bool = True,
                 method: str = "spectralclustering",
                 n_leaflets: int = 2,
                 normal_axis: str = "z",
                 add_topologyattribute: bool = False,
                 **kwargs):
        self._cache = {}
        self.universe = universe.universe
        self.pbc = pbc
        self.n_leaflets = n_leaflets
        self.cutoff = cutoff
        self.kwargs = dict(**kwargs)
        self._normal_axis = ["x", "y", "z"].index(normal_axis)

        self.atomgroup = universe.select_atoms(select, periodic=pbc)
        self.atoms_by_residue = self.atomgroup.split("residue")
        self._first_residue_atoms = sum(ag[0] for ag in self.atoms_by_residue)
        self.residues = self.atomgroup.residues
        self.n_residues = len(self.residues)
        if select_tailgroups is not None:
            self.tailgroups = self.residues.atoms.select_atoms(select_tailgroups,
                                                               periodic=pbc)
        else:
            self.tailgroups = self.residues.atoms - self.atomgroup

        if pbc:
            self._get_box = lambda: self.universe.dimensions
        else:
            self._get_box = lambda: None

        if isinstance(method, str):
            method = method.lower().replace('_', '')
        if method == "graph":
            self.method = GraphMethod(self.atomgroup, self.tailgroups,
                                      cutoff=self.cutoff, pbc=self.pbc,
                                      **kwargs)
            self._method = self.method.run
        elif method == "spectralclustering":
            self.method = SpectralClusteringMethod(self.atomgroup, self.tailgroups,
                                                   cutoff=self.cutoff, pbc=self.pbc,
                                                   n_leaflets=self.n_leaflets,
                                                   **kwargs)
            self._method = self.method.run
        else:
            self._method = self.method = method

    @property
    def box(self):
        return self._get_box()

    def run(self):
        """
        This clears the cache for lazy running.
        """
        self._cache = {}
        self._output_leaflet_indices

    def write_selection(self, filename, mode="w", format=None, **kwargs):
        """Write selections for the leaflets to *filename*.

        The format is typically determined by the extension of *filename*
        (e.g. "vmd", "pml", or "ndx" for VMD, PyMol, or Gromacs).

        See :class:`MDAnalysis.selections.base.SelectionWriter` for all
        options.
        """
        sw = get_writer(filename, format)
        with sw(filename, mode=mode,
                preamble=f"Leaflets found by {repr(self)}\n",
                **kwargs) as writer:
            for i, ag in enumerate(self.leaflet_atoms, 1):
                writer.write(ag, name=f"leaflet_{i:d}")

    def __repr__(self):
        return (f"LeafletFinder(method={self.method}, select='{self.atomgroup}', "
                f"cutoff={self.cutoff:.1f} Å, pbc={self.pbc})")

    @cached_property
    def _output_leaflet_indices(self):
        clusters = self._method(selection=self.atomgroup,
                                tailgroups=self.tailgroups,
                                cutoff=self.cutoff, box=self.box,
                                **self.kwargs)
        return [sorted(x) for x in clusters]

    @cached_property
    def residue_leaflets(self):
        arr = np.full(self.n_residues, -1, dtype=int)
        for leaflet, residues in enumerate(self._output_leaflet_indices):
            for residue_index in residues:
                arr[residue_index] = leaflet
        return arr

    @cached_property
    def _output_leaflet_residues(self):
        return [self.residues[x] for x in self._output_leaflet_indices]

    def _get_atomgroup_by_indices(self, indices):
        ag = sum(self.atoms_by_residue[i] for i in indices)
        if not ag:
            return self.atomgroup[[]]
        return ag

    @cached_property
    def _output_leaflet_atoms(self):
        return [self._get_atomgroup_by_indices(x)
                for x in self._output_leaflet_indices]

    @cached_property
    def leaflet_indices_by_size(self):
        return sorted(self._output_leaflet_indices, key=len, reverse=True)

    @cached_property
    def leaflet_residues_by_size(self):
        return [self.residues[x] for x in self.leaflet_indices_by_size]

    @cached_property
    def leaflet_atoms_by_size(self):
        return [self._get_atomgroup_by_indices(x)
                for x in self.leaflet_indices_by_size]

    def _argsort_by_normal(self, groups):
        vals = [np.mean(x.positions[:, self._normal_axis]) for x in groups]
        return np.argsort(vals)[::-1]

    @cached_property
    def _output_by_normal(self):
        return self._argsort_by_normal(self._output_leaflet_atoms)

    @cached_property
    def leaflet_indices_by_normal(self):
        return [self._output_leaflet_indices[i]
                for i in self._output_by_normal]

    @cached_property
    def leaflet_residues_by_normal(self):
        return [self._output_leaflet_residues[i]
                for i in self._output_by_normal]

    @cached_property
    def leaflet_atoms_by_normal(self):
        return [self._output_leaflet_atoms[i]
                for i in self._output_by_normal]

    @cached_property
    def _argsort_by_size_and_normal(self):
        atoms = self.leaflet_atoms_by_size[:self.n_leaflets]
        return self._argsort_by_normal(atoms)

    @cached_property
    def leaflet_indices(self):
        return [self.leaflet_indices_by_size[i]
                for i in self._argsort_by_size_and_normal]

    @cached_property
    def leaflet_residues(self):
        return [self.leaflet_residues_by_size[i]
                for i in self._argsort_by_size_and_normal]

    @cached_property
    def leaflet_atoms(self):
        return [self.leaflet_atoms_by_size[i]
                for i in self._argsort_by_size_and_normal]

    @cached_property
    def resindex_to_leaflet(self):
        r2l = {}
        for i, residues in enumerate(self.leaflet_residues):
            for resindex in residues.resindices:
                self.resindex_to_leaflet[resindex] = i

    def atom_leaflet_by_distance(self, atom):
        pairs = capped_distance(atom.position,
                                self._first_residue_atoms.positions,
                                max_cutoff=self.cutoff,
                                box=self.box, return_distances=False)
        neighbors = self.residue_leaflets[pairs[:, 1]]
        most_common = np.bincount(neighbors).argmax()
        return most_common

    def assign_atoms_by_distance(self, atomgroup):
        leaflets = np.full(len(atomgroup), -1, dtype=int)
        for i, atom in enumerate(atomgroup):
            leaflets[i] = self.atom_leaflet_by_distance(atom)
        return leaflets

    def get_first_outside_atoms(self, residues):
        atoms = residues.atoms.split("residue")
        residues = residues.residues
        outside_ix = np.where(~np.in1d(residues, self.residues))[0]
        if not len(outside_ix):
            return residues.atoms[[]]
        outside_atoms = sum([atoms[i][0] for i in outside_ix])
        return outside_atoms
