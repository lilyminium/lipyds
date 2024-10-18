"""
Grouping methods
================

Classes
-------

.. autoclass:: GraphMethod
    :members:

.. autoclass:: SpectralClusteringMethod
    :members:
"""

import abc
import warnings
from typing import Optional, List

from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.analysis.distances import contact_matrix
import numpy as np
from numpy.typing import ArrayLike

from ..lib.mdautils import (get_centers_by_residue, get_orientations,
                            get_distances_with_projection)


class GroupingMethod(abc.ABC):
    def __init__(self, headgroups: AtomGroup,
                 tailgroups: Optional[AtomGroup]=None,
                 cutoff: float=15.0, pbc: bool=False,
                 n_leaflets: int = 2,
                 ):
        self.headgroups = headgroups
        if tailgroups is None:
            tailgroups = self.headgroups.residues.atoms - headgroups
        self.tailgroups = tailgroups
        self.n_residues = len(self.headgroups.residues)
        self.cutoff = cutoff
        self.n_leaflets = n_leaflets
        self.pbc = pbc
        if pbc:
            self.get_box = lambda: self.headgroups.universe.dimensions
        else:
            self.get_box = lambda: None

    @abc.abstractmethod
    def run(self, **kwargs) -> List[List[int]]:
        raise NotImplementedError()

class GraphMethod(GroupingMethod):

    name = "graph"

    def __init__(self, headgroups: AtomGroup,
                 tailgroups: Optional[AtomGroup]=None,
                 cutoff: float=15.0, pbc: bool=False,
                 sparse: Optional[bool]=None,
                 n_leaflets: int = 2,
                 **kwargs
                 ):
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("networkx is required to use this method "
                            "but is not installed. Install it with "
                            "`conda install networkx` or "
                            "`pip install networkx`.") from None
        super().__init__(headgroups, tailgroups=tailgroups,
                         cutoff=cutoff, pbc=pbc,
                        n_leaflets=n_leaflets)
        self.sparse = sparse
        self.returntype = "numpy" if not sparse else "sparse"
        self.graph = None

    def run(self, **kwargs) -> List[List[int]]:
        import networkx as nx
        box = self.get_box()
        coordinates = get_centers_by_residue(self.headgroups, box=box)
        try:
            adj = contact_matrix(coordinates, cutoff=self.cutoff, box=box,
                                returntype=self.returntype)
        except ValueError as exc:
            if self.sparse is None:
                warnings.warn("NxN matrix is too big. Switching to sparse "
                            "matrix method")
                adj = contact_matrix(coordinates, cutoff=self.cutoff, box=box,
                                    returntype="sparse")
            elif self.sparse is False:
                raise ValueError("NxN matrix is too big. "
                                "Use `sparse=True`") from None
            else:
                raise exc 
        self.graph = nx.Graph(adj)
        groups = [list(c) for c in nx.connected_components(self.graph)]
        return groups[:self.n_leaflets]


class SpectralClusteringMethod(GroupingMethod):

    name = "spectralclustering"

    def __init__(self, headgroups: AtomGroup,
                 tailgroups: Optional[AtomGroup]=None,
                 cutoff: float=30.0, pbc: bool=False,
                 n_leaflets: int=2, delta: Optional[float]=20,
                 cosine_threshold: float=0.8,
                 angle_factor: float=1, **kwargs):
        
        try:
            import sklearn.cluster as skc
        except ImportError:
            raise ImportError('scikit-learn is required to use this method '
                            'but is not installed. Install it with `conda '
                            'install scikit-learn` or `pip install '
                            'scikit-learn`.') from None

        super().__init__(headgroups, tailgroups=tailgroups,
                         cutoff=cutoff, pbc=pbc, n_leaflets=n_leaflets)
        self.delta = delta
        self.cosine_threshold = cosine_threshold
        self.angle_factor = angle_factor
        self.predictor = skc.SpectralClustering(n_clusters=n_leaflets,
                                                affinity="precomputed",
                                                **kwargs)
    
    def _get_kernel(self) -> ArrayLike:
        box = self.get_box()
        
        coordinates = get_centers_by_residue(self.headgroups, box=box)
        orientations = get_orientations(self.headgroups,
                                        tailgroups=self.tailgroups,
                                        box=box, normalize=True,
                                        headgroup_centers=coordinates)
        dist_mat = get_distances_with_projection(coordinates, orientations,
                                                 self.cutoff, box=box,
                                                 angle_factor=self.angle_factor)
        delta = self.delta or np.max(dist_mat[dist_mat < self.cutoff*2]) / 3

        gau = np.exp(- dist_mat ** 2 / (2. * delta ** 2))
        # reasonably acute/obtuse angles are acute/obtuse anough
        angles = np.dot(orientations, orientations.T)
        cos = np.clip(angles, -self.cosine_threshold, self.cosine_threshold)
        cos += self.cosine_threshold
        cos /= (2*self.cosine_threshold)
        mask = ~np.isnan(cos)
        gau[mask] *= cos[mask]
        return gau

    def run(self, **kwargs) -> List[List[int]]:
        kernel = self._get_kernel()
        data_labels = self.predictor.fit_predict(kernel)
        ix = np.argsort(data_labels)
        indices = np.arange(self.n_residues)
        splix = np.where(np.ediff1d(data_labels[ix]))[0] + 1
        cluster_indices = np.split(indices[ix], splix)
        return cluster_indices[:self.n_leaflets]