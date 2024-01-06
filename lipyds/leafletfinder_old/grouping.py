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
import typing

from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.analysis.distances import contact_matrix
import numpy as np
from numpy.typing import ArrayLike

from ..lib.mdautils import (get_centers_by_residue, get_orientations,
                            get_distances_with_projection)


class GroupingMethod(abc.ABC):
    def __init__(self, leafletfinder):
        self._leafletfinder = leafletfinder

    @abc.abstractmethod
    def run(self, **kwargs) -> list[list[int]]:
        """
        This method should return a list of lists of indices.
        Each list contains the relative indices of the residues in a leaflet.
        """
        raise NotImplementedError()
    
    @property
    def leafletfinder(self):
        return self._leafletfinder
    
    @property
    def _unwrapped_headgroup_centers(self):
        return self.leafletfinder.lipids.unwrapped_headgroup_centers

class GraphMethod(GroupingMethod):

    def __init__(self, leafletfinder, sparse: bool = False,):
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("networkx is required to use this method "
                            "but is not installed. Install it with "
                            "`conda install networkx` or "
                            "`pip install networkx`.") from None
        super().__init__(leafletfinder)
        self.sparse = sparse
        self.returntype = "numpy" if not sparse else "sparse"

    def run(self, **kwargs) -> list[list[int]]:
        import networkx as nx

        coordinates = self._unwrapped_headgroup_centers
        try:
            adj = contact_matrix(
                coordinates,
                cutoff=self.leafletfinder.cutoff,
                box=self.leafletfinder.box,
                returntype=self.returntype,
            )
        except ValueError as exc:
            if self.sparse is None:
                warnings.warn("NxN matrix is too big. Switching to sparse "
                            "matrix method")
                adj = contact_matrix(
                    coordinates,
                    cutoff=self.leafletfinder.cutoff,
                    box=self.leafletfinder.box,
                    returntype="sparse",
                )
            elif self.sparse is False:
                raise ValueError("NxN matrix is too big. "
                                "Use `sparse=True`") from None
            else:
                raise exc 
        graph = nx.Graph(adj)
        groups = sorted(
            [list(c) for c in nx.connected_components(graph)],
            key=len,
            reverse=True,
        )[:self.leafletfinder.n_leaflets]
        return groups


class SpectralClusteringMethod(GroupingMethod):
    def __init__(
        self,
        leafletfinder,
        delta: typing.Optional[float] = 20,
        cosine_threshold: float = 0.8,
        angle_factor: float = 1,
        **kwargs
    ):
        try:
            import sklearn.cluster as skc
        except ImportError:
            raise ImportError('scikit-learn is required to use this method '
                            'but is not installed. Install it with `conda '
                            'install scikit-learn` or `pip install '
                            'scikit-learn`.') from None

        super().__init__(leafletfinder)
        self.delta = delta
        self.cosine_threshold = cosine_threshold
        self.angle_factor = angle_factor
        self.predictor = skc.SpectralClustering(
            n_clusters=self.leafletfinder.n_leaflets,
            affinity="precomputed",
            **kwargs,
        )
    
    def _get_kernel(self) -> ArrayLike:
        coordinates = self._unwrapped_headgroup_centers
        orientations = np.array([
            lipid.normalized_orientation
            for lipid in self._leafletfinder.lipids
        ])
        
        dist_mat = get_distances_with_projection(
            coordinates,
            orientations,
            self.leafletfinder.cutoff,
            box=self.leafletfinder.box,
            angle_factor=self.angle_factor,
        )
        delta = (
            self.delta
            or np.max(dist_mat[dist_mat < self.leafletfinder.cutoff * 2]) / 3
        )

        gau = np.exp(- dist_mat ** 2 / (2. * delta ** 2))
        # reasonably acute/obtuse angles are acute/obtuse anough
        angles = np.dot(orientations, orientations.T)
        cos = np.clip(angles, -self.cosine_threshold, self.cosine_threshold)
        cos += self.cosine_threshold
        cos /= (2*self.cosine_threshold)
        mask = ~np.isnan(cos)
        gau[mask] *= cos[mask]
        return gau

    def run(self, **kwargs) -> list[list[int]]:
        kernel = self._get_kernel()
        data_labels = self.predictor.fit_predict(kernel)
        ix = np.argsort(data_labels)
        indices = np.arange(self.n_residues)
        splix = np.where(np.ediff1d(data_labels[ix]))[0] + 1
        cluster_indices = np.split(indices[ix], splix)
        return cluster_indices