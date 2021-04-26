from typing import Optional, Union

import numpy as np
from MDAnalysis.core.universe import Universe
from MDAnalysis.core.groups import AtomGroup

from .grouping import GraphMethod, SpectralClusteringMethod
from .utils import get_centers_by_residue



class LeafletFinder:
    """Identify atoms in the same leaflet of a lipid bilayer.

    You can use a predefined method ("graph", "spectralclustering" or
    "center_of_geometry"). Alternatively, you can pass in your own function
    as a method. This *must* accept an array of coordinates as the first
    argument, and *must* return either a list of numpy arrays (the
    ``components`` attribute) or a tuple of (list of numpy arrays,
    predictor object). The numpy arrays should be arrays of indices of the
    input coordinates, such that ``k = components[i][j]`` means that the
    ``k``th coordinate belongs to the ``i-th`` leaflet.
    The class will also pass the following keyword arguments to your function:
    ``cutoff``, ``box``, ``return_predictor``.

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
        "graph" for :func:`~distances.group_coordinates_by_graph`;
        "spectralclustering" for
        :func:`~distances.group_coordinates_by_spectralclustering`;
        "center_of_geometry" for
        :func:`~distances.group_coordinates_by_cog`;
        "orientation" to calculate orientations for each lipid and
        use :func:`~distances.group_vectors_by_orientation`
        or alternatively, pass in your own method. This *must* accept an
        array of coordinates as the first argument, and *must*
        return either a list of numpy arrays (the ``components``
        attribute) or a tuple of (list of numpy arrays, predictor object).
    calculate_orientations: bool (optional)
        if your custom method requires the orientation vector of each lipid,
        set ``calculate_orientations=True`` and an Nx3 array of orientation
        vectors will get passed into your function with the keyword
        ``orientation``. This is set to ``True`` for ``method="orientation"``.
    **kwargs:
        Passed to ``method``


    Attributes
    ----------
    universe: Universe
    select: str
        Selection string
    selection: AtomGroup
        Atoms that the analysis is applied to
    headgroups: List of AtomGroup
        Atoms that the analysis is applied to, grouped by residue.
    pbc: bool
        Whether to use PBC or not
    box: numpy.ndarray or None
        Cell dimensions to use in calculating distances
    predictor:
        The object used to group the leaflets. :class:`networkx.Graph` for
        ``method="graph"``; :class:`sklearn.cluster.SpectralClustering` for
        ``method="spectralclustering"``; or :class:`numpy.ndarray` for
        ``method="center_of_geometry"``.
    positions: numpy.ndarray (N x 3)
        Array of positions of headgroups to use. If your selection has
        multiple atoms for each residue, this is the center of geometry.
    orientations: numpy.ndarray (N x 3) or None
        Array of orientation vectors calculated with ``lipid_orientation``.
    components: list of numpy.ndarray
        List of indices of atoms in each leaflet, corresponding to the
        order of `selection`. ``components[i]`` is the array of indices
        for the ``i``-th leaflet. ``k = components[i][j]`` means that the
        ``k``-th atom in `selection` is in the ``i``-th leaflet.
        The components are sorted by size for the "spectralclustering" and
        "graph" methods. For the "center_of_geometry" method, they are
        sorted by the order that the centers are passed into the class.
    groups: list of AtomGroups
        List of AtomGroups in each leaflet. ``groups[i]`` is the ``i``-th
        leaflet. The components are sorted by size for the "spectralclustering"
        and "graph" methods. For the "center_of_geometry" method, they are
        sorted by the order that the centers are passed into the class.
    leaflets: list of AtomGroup
        List of AtomGroups in each leaflet. ``groups[i]`` is the ``i``-th
        leaflet. The leaflets are sorted by z-coordinate so that the
        upper-most leaflet is first.
    sizes: list of ints
        List of the size of each leaflet in ``groups``.


    Example
    -------
    The components of the graph are stored in the list
    :attr:`LeafletFinder.components`; the atoms in each component are numbered
    consecutively, starting at 0. To obtain the atoms in the input structure
    use :attr:`LeafletFinder.groups`::

       L = LeafletFinder(PDB, 'name P*')
       leaflet_1 = L.groups[0]
       leaflet_2 = L.groups[1]

    The residues can be accessed through the standard MDAnalysis mechanism::

       leaflet_1.residues

    provides a :class:`~MDAnalysis.core.groups.ResidueGroup`
    instance. Similarly, all atoms in the first leaflet are then ::

       leaflet_1.residues.atoms


    See also
    --------
    :func:`~MDAnalysis.analysis.distances.group_coordinates_by_graph`
    :func:`~MDAnalysis.analysis.distances.group_coordinates_by_spectralclustering`


    .. versionchanged:: 2.0.0
        Refactored to move grouping code into ``distances`` and use
        multiple methods. Added the "spectralclustering" and
        "center_of_geometry" methods.

    .. versionchanged:: 1.0.0
       Changed `selection` keyword to `select`
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
        self.leaflet_indices = [sorted(x) for x in clusters]
        self.leaflet_residues = [self.residues[x]
                                 for x in self.leaflet_indices]
        self.leaflet_atoms = [x.atoms for x in self.leaflet_residues]

        if self.order_leaflets_by_size:
            self.leaflet_indices_by_size = sorted(self.leaflet_indices,
                                                  key=len, reverse=True)
            self.leaflet_residues_by_size = [self.residues[x] for x in
                                             self.leaflet_indices_by_size]
            self.leaflet_atoms_by_size = [x.atoms for x in
                                          self.leaflet_residues_by_size]
        
        if self.order_leaflets_by_z:
            zs = [np.mean(x.positions[:, 2]) for x in self.leaflet_atoms]
            order = np.argsort(zs)[::-1]

            self.leaflet_indices_by_z = [self.leaflet_indices[x] for x in order]
            self.leaflet_residues_by_z = [self.leaflet_residues[x]
                                          for x in order]
            self.leaflet_atoms_by_z = [self.leaflet_atoms[x] for x in order]

        
    def __init__(self, universe: Union[AtomGroup, Universe],
                 select: Optional[str]='all',
                 select_tailgroups: Optional[str]=None,
                 cutoff: float=40,
                 pbc: bool=True,
                 method: str="spectralclustering",
                 order_leaflets_by_size: bool=True,
                 order_leaflets_by_z: bool=True,
                 **kwargs):
        self.universe = universe.universe
        self.pbc = pbc
        self.selection = universe.select_atoms(select, periodic=pbc)
        self.sel_by_residue = self.selection.split("residue")
        self.residues = self.selection.residues
        self.n_residues = len(self.residues)
        if select_tailgroups is not None:
            self.tailgroups = self.residues.atoms.select_atoms(select_tailgroups,
                                                               periodic=pbc)
        else:
            self.tailgroups = self.residues.atoms - self.selection

        self.cutoff = cutoff
        self.order_leaflets_by_size = order_leaflets_by_size
        self.order_leaflets_by_z = order_leaflets_by_z
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
                                                   **kwargs)
            self._method = self.method.run
        else:
            self._method = self.method = method
        self.run()

    def groups_iter(self):
        """Iterator over all leaflet :meth:`groups`"""
        for group in self.groups:
            yield group

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
            for i, ag in enumerate(self.groups, 1):
                writer.write(ag, name=f"leaflet_{i:d}")

    def __repr__(self):
        return (f"LeafletFinder(method={self.method}, select='{self.select}', "
                f"cutoff={self.cutoff:.1f} Å, pbc={self.pbc})")



def optimize_cutoff(universe, select, dmin=10.0, dmax=20.0, step=0.5,
                    max_imbalance=0.2, **kwargs):
    r"""Find cutoff that minimizes number of disconnected groups.

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
    universe : Universe
        :class:`MDAnalysis.Universe` instance
    select : AtomGroup or str
        AtomGroup or selection string as used for :class:`LeafletFinder`
    dmin : float (optional)
    dmax : float (optional)
    step : float (optional)
        scan cutoffs from `dmin` to `dmax` at stepsize `step` (in Angstroms)
    max_imbalance : float (optional)
        tuning parameter for the balancing heuristic [0.2]
    kwargs : other keyword arguments
        other arguments for  :class:`LeafletFinder`

    Returns
    -------
    (cutoff, N)
         optimum cutoff and number of groups found


    .. Note:: This function can die in various ways if really no
              appropriate number of groups can be found; it ought  to be
              made more robust.

    .. versionchanged:: 1.0.0
       Changed `selection` keyword to `select`
    """
    kwargs.pop('cutoff', None)  # not used, so we filter it
    _sizes = []
    for cutoff in np.arange(dmin, dmax, step):
        LF = LeafletFinder(universe, select, cutoff=cutoff, method="graph", **kwargs)
        # heuristic:
        #  1) N > 1
        #  2) no imbalance between large groups:
        sizes = LF.sizes
        if len(sizes) < 2:
            continue
        n0 = float(sizes[0])  # sizes of two biggest groups ...
        n1 = float(sizes[1])  # ... assumed to be the leaflets
        imbalance = np.abs(n0 - n1) / (n0 + n1)
        # print "sizes: %(sizes)r; imbalance=%(imbalance)f" % vars()
        if imbalance > max_imbalance:
            continue
        _sizes.append((cutoff, len(LF.sizes)))
    results = np.rec.fromrecords(_sizes, names="cutoff,N")
    del _sizes
    results.sort(order=["N", "cutoff"])  # sort ascending by N, then cutoff
    return results[0]  # (cutoff,N) with N>1 and shortest cutoff