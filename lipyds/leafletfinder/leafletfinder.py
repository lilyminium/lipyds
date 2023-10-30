"""
LeafletFinder
=============

Classes
-------

.. autoclass:: LeafletFinder
    :members:
"""

import typing

import numpy as np
from MDAnalysis.core.universe import Universe
from MDAnalysis.core.groups import AtomGroup

from lipyds.core.groups import LipidGroup

from .grouping import GroupingMethod, GraphMethod, SpectralClusteringMethod
from ..lib.utils import cached_property


class LeafletFinder:
    """
    Partition residues into leaflets.

    Parameters
    ----------
    universe_or_atomgroup: Universe or AtomGroup
        Atoms to apply the algorithm to
    select: str
        Selection string for headgroup atoms to apply the algorithm to.
        Multiple atoms can be selected for each residue,
        in which case the center of geometry of the headgroup
        atoms is used.
    select_tailgroups: str, optional
        Selection string for tailgroup atoms to apply the algorithm to.
        If not given, all atoms that are not in ``select`` are used.
    normal_axis: {'x', 'y', 'z'}, optional
        The normal axis of the membrane.
        This is used to sort the leaflets by their position along the axis.
        The default is ``'z'``.
    cutoff: float, optional
        The cutoff for the clustering algorithm.
        This is generally used for selecting neighbours.
    method: {'graph', 'spectralclustering'}, optional
        The method to use for clustering.
        This can also be an instance of GroupingMethod.
    pbc: bool, optional
        Whether to use periodic boundary conditions in leaflet assignment.
    n_leaflets: int, optional
        The number of leaflets to return.
        The default is 2.
    update_TopologyAttr: bool, optional
        Whether to update the ``leaflet`` attribute of the residues.

    Attributes
    ----------
    lipids: LipidGroup
        The lipids in the system.
    residues: mda.core.groups.ResidueGroup
        The residues in the system.
    cutoff: float
        The cutoff for the clustering algorithm.
        This is generally used for selecting neighbours.
    normal_axis: {'x', 'y', 'z'}
        The normal axis of the membrane.
        This is used to sort the leaflets by their position along the axis.
    box: np.ndarray
        The box of the system.
    leaflet_local_indices: list[np.ndarray]
        The local indices of the residues in each leaflet.
        leaflet_local_indices[leaflet_index, residue_index]
        shows that the ``LeafletFinder.residues[residues_index]`` residue
        is in the ``leaflet_index`` leaflet.
    residue_leaflet_indices: np.ndarray
        The leaflet integer index of each residue.
        -1 stands for no leaflet; 0 for the first leaflet, 1 for the second, etc.
        Leaflets are ordered by their position along the normal axis.
        This is of shape (n_residues,).
        The ``LeafletFinder.residues[residue_index]`` residue is in the
        ``residue_leaflet_indices[residue_index]`` leaflet.
    """

    def __init__(
        self,
        universe_or_atomgroup: typing.Union[AtomGroup, Universe],
        select: str = "all",
        select_tailgroups: typing.Optional[str] = None,
        normal_axis: typing.Literal["x", "y", "z"] = "z",
        cutoff: float = 40,
        method: typing.Literal["graph", "spectralclustering"] = "graph",
        pbc: bool = True,
        n_leaflets: int = 2,
        update_TopologyAttr: bool = True,
        **kwargs
    ):
        self._cache = {}
        self._universe = universe_or_atomgroup.universe
        self._lipids = LipidGroup.from_atom_selections(
            universe_or_atomgroup,
            select_headgroups=select,
            select_tailgroups=select_tailgroups,
        )
        self.residues = universe_or_atomgroup.select_atoms(select).residues
        self._first_atoms = sum([
            lipid._first_headgroup_atom
            for lipid in self._lipids
        ])

        self.cutoff = cutoff
        self.n_leaflets = n_leaflets
        self.normal_axis = normal_axis
        self._normal_axis_index = ["x", "y", "z"].index(normal_axis.lower())

        if pbc:
            self._get_box = lambda: self._universe.dimensions
        else:
            self._get_box = lambda: None

        SUPPORTED_METHODS = {
            "graph": GraphMethod,
            "spectralclustering": SpectralClusteringMethod,
        }
        if isinstance(method, str):
            method = method.lower().replace("_", "")
            try:
                self.method = SUPPORTED_METHODS[method](self, **kwargs)
            except KeyError:
                raise ValueError(
                    f"Method {method} is not supported. "
                    f"Supported methods are: {list(SUPPORTED_METHODS)}"
                ) from None
            else:
                self._method = self.method.run
        elif isinstance(method, GroupingMethod):
            self.method = method(self, **kwargs)
            self._method = self.method.run
        else:
            self._method = self.method = method

        self._update_TopologyAttr = update_TopologyAttr
        if update_TopologyAttr:
            self._universe.add_TopologyAttr("leaflet")

    @property
    def box(self):
        return self._get_box()

    @property
    def lipids(self):
        return self._lipids
    
    @cached_property
    def leaflet_local_indices(self):
        return self._run()
    
    @cached_property
    def residue_leaflet_indices(self):
        arr = np.full(self.residues.n_residues, -1, dtype=int)
        for leaflet_index, residue_indices in enumerate(self.leaflet_local_indices):
            for residue_index in residue_indices:
                arr[residue_index] = leaflet_index
        return arr
    
    @cached_property
    def leaflets(self):
        from lipyds.core.groups import Leaflet
        return[
            Leaflet(self.lipids[cluster])
            for cluster in self.leaflet_local_indices
        ]
    
    def get_nearest_leaflet_index(
        self,
        position: np.ndarray,
        cutoff: float = 10,
    ) -> int:
        """
        Get the index of the nearest leaflet to a position.
        This method searches all *first* headgroup positions within
        ``cutoff`` of the position. It returns the leaflet index
        of the leaflet with the *most* headgroups within ``cutoff``.

        Parameters
        ----------
        position: np.ndarray
            The position to search from. Should have shape (1, 3)
        cutoff: float, optional
            The cutoff to search within (Angstrom)
        
        Returns
        -------
        int
            The index of the nearest leaflet.
            -1 if no leaflet is within ``cutoff``.
        """
        from MDAnalysis.analysis.distances import capped_distance

        target = position.reshape((1, 3))
        references = self.lipids._first_headgroup_atoms.positions

        neighbors = capped_distance(
            target,
            references,
            cutoff=cutoff,
            box=self.box,
            return_distances=False,
        ).T
        if not len(neighbors):
            return -1
        
        # count the number of neighbors in each leaflet
        leaflet_indices = self.residue_leaflet_indices[neighbors]
        most_common = np.bincount(leaflet_indices).argmax()
        return most_common
        
        
    
    def run(self):
        """Run the leaflet finding algorithm and update properties"""
        self._cache = {}
        self.leaflet_local_indices
        if self._update_TopologyAttr:
            self.residues.leaflets = self.residue_leaflet_indices
    
    def _run(self) -> list[np.ndarray]:
        from lipyds.lib.mdautils import unwrap_coordinates

        cluster_indices: list[np.ndarray] = [
            np.asarray(cluster)
            for cluster in self._method()
        ]
        positions = self._first_atoms.positions
        unwrapped = unwrap_coordinates(positions, positions[0], self.box)

        axis_values = [
            unwrapped[cluster, self._normal_axis_index].mean()
            for cluster in cluster_indices
        ]
        argsort = np.argsort(axis_values)[::-1]
        return [cluster_indices[i] for i in argsort]
    
    