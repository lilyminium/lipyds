"""
LeafletFinder
=============


Classes
-------

.. autoclass:: LeafletFinder
    :members:
"""

import itertools
from typing import Optional, Union

import numpy as np
from MDAnalysis.core.universe import Universe
from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.selections import get_writer
from MDAnalysis.analysis.distances import capped_distance, distance_array

from .grouping import GraphMethod, SpectralClusteringMethod
from ..lib.utils import cached_property
from ..lib import mdautils


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
                 cutoff: float = 15,
                 pbc: bool = True,
                 method: str = "graph",
                 n_leaflets: int = 2,
                 normal_axis: str = "z",
                 update_TopologyAttr: bool = False,
                 **kwargs):
        self._cache = {}
        self.universe = universe.universe
        self.pbc = pbc
        self.n_leaflets = n_leaflets
        self._cutoff = cutoff
        self.kwargs = dict(**kwargs)
        self._normal_axis = ["x", "y", "z"].index(normal_axis)
        self._select = select

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
                                      n_leaflets=self.n_leaflets,
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

        self._update_TopologyAttr = update_TopologyAttr

    @property
    def cutoff(self):
        return self._cutoff

    @cutoff.setter
    def cutoff(self, value):
        self._cutoff = value
        self.method.cutoff = value

    @property
    def box(self):
        return self._get_box()

    def run(self):
        """
        This clears the cache for lazy running.
        """
        self._cache = {}
        self._output_leaflet_indices
        if self._update_TopologyAttr:
            self.atomgroup.universe.add_TopologyAttr("leaflet")
            for i, residues in enumerate(self.leaflet_residues):
                for residue in residues:
                    residue.leaflet = i

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
        try:
            name = self.method.name
        except AttributeError:
            name = self.method
        return (f"LeafletFinder(method={name}, select='{self._select}', "
                f"cutoff={self.cutoff:.1f} Å, pbc={self.pbc})")
    
    def _run(self):
        return self._method(selection=self.atomgroup,
                            tailgroups=self.tailgroups,
                            cutoff=self.cutoff, box=self.box,
                            **self.kwargs)

    @cached_property
    def _output_leaflet_indices(self):
        clusters = self._run()
        return [sorted(x) for x in clusters]

    @cached_property
    def residue_leaflets(self):
        arr = np.full(self.n_residues, -1, dtype=int)
        for leaflet, residues in enumerate(self.leaflet_indices):
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
        positions = [x.positions for x in groups]
        unwrapped = [mdautils.unwrap_coordinates(x, x[0], self.box) for x in positions]
        vals = [np.mean(x[:, self._normal_axis]) for x in unwrapped]
        args = np.argsort(vals)[::-1]
        return args

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
    def leaflet_coordinates(self):
        by_leaflet = [mdautils.get_centers_by_residue(ag, box=self.box)
                      for ag in self.leaflet_atoms]
        unwrapped = [mdautils.unwrap_coordinates(x, center=by_leaflet[0][0], box=self.box)
                     for x in by_leaflet]
        center = np.concatenate(unwrapped).mean(axis=0)
        return unwrapped

    @cached_property
    def resindex_to_leaflet(self):
        r2l = {}
        for i, residues in enumerate(self.leaflet_residues):
            for resindex in residues.resindices:
                self.resindex_to_leaflet[resindex] = i

    def atom_leaflet_by_distance(self, atom, cutoff=10):
        zs = self._first_residue_atoms.positions
        # zs[:, :2] = 0

        atom_z = atom.position
        # atom_z[:2] = 0
        pairs, dists = capped_distance(atom_z,
                                        zs,
                                        max_cutoff=self.cutoff,
                                        box=self.box, return_distances=True)
        if dists.min() > cutoff:
            return -1
        if len(pairs):
            neighbors = self.residue_leaflets[pairs[:, 1]]
            most_common = np.bincount(neighbors).argmax()
            return most_common
        distances = distance_array(atom.position,
                                   self._first_residue_atoms.positions,
                                   box=self.box).reshape(-1)
        arg = distances.argmin()
        return self.residue_leaflets[distances.argmin()]

    def assign_atoms_by_distance(self, atomgroup, cutoff=10):
        leaflets = np.full(len(atomgroup), -1, dtype=int)
        for i, atom in enumerate(atomgroup):
            leaflets[i] = self.atom_leaflet_by_distance(atom, cutoff=cutoff)
        return leaflets

    def get_first_outside_atoms(self, residues):
        atoms = residues.atoms.split("residue")
        residues = residues.residues
        outside_ix = np.where(~np.in1d(residues, self.residues))[0]
        if not len(outside_ix):
            return residues.atoms[[]]
        outside_atoms = sum([atoms[i][0] for i in outside_ix])
        return outside_atoms


    @classmethod
    def optimize_cutoff(
        cls,
        universe,
        dmin: float = 10.0,
        dmax: float = 20.0,
        step: float = 0.5,
        max_imbalance: float = 0.2,
        n_leaflets: int = 2,
        **kwargs
    ) -> float:
        """
        Find cutoff that minimizes number of disconnected groups.

        Applies heuristics to find best groups:

            1. at least two groups (assumes that there are at least 2 leaflets)
            2. reject any solutions for which:

        .. math::

                \frac{|N_0 - N_1|}{|N_0 + N_1|} > \mathrm{max_imbalance}

        with :math:`N_i` being the number of lipids in group
        :math:`i`. This heuristic picks groups with balanced numbers of
        lipids.

        
        Parameters
        ----------
        universe: Universe or AtomGroup
            Atoms to apply the algorithm to
        dmin: float, optional
            Minimum cutoff distance
        dmax: float, optional
            Maximum cutoff distance
        step: float, optional
            Step size for searching cutoff distances
        max_imbalance: float, optional
            Maximum imbalance between groups allowed
        n_leaflets: int, optional
            Number of leaflets to find
        **kwargs:
            Passed to :class:`LeafletFinder`
        """

        def _run(cutoff):
            lf = cls(universe, cutoff=cutoff, n_leaflets=n_leaflets, **kwargs)
            lf.run()
            return lf
        
        valid_cutoffs = []
        for cutoff in np.arange(dmin, dmax, step):
            lf = _run(cutoff)

            # if we don't have enough leaflets, continue
            if len(lf._output_leaflet_indices) < n_leaflets:
                continue

            # check imbalance between groups
            sizes = [len(x) for x in lf.leaflet_residues_by_size]
            for i, j in itertools.combinations(range(n_leaflets), 2):
                upper = abs(sizes[i] - sizes[j])
                lower = sizes[i] + sizes[j]
                imbalance = upper / lower
                if imbalance > max_imbalance:
                    continue
            valid_cutoffs.append((-sum(sizes), cutoff))
        
        if not valid_cutoffs:
            raise ValueError("No valid cutoffs found.")
            
        # sort so first item is highest number of lipids matched, lowest cutoff
        valid_cutoffs.sort()
        return valid_cutoffs[0][-1]

        
