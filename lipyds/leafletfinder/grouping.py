
import abc
import typing
import warnings

import numpy as np

from lipyds.core.groups import LipidGroup, Leaflet

class GroupingMethod(abc.ABC):

    @abc.abstractmethod
    def _run(self, lipids: LipidGroup) -> list[list[int]]:
        """
        This method should return a list of lists of indices.
        Each list contains the relative indices of the residues in a leaflet.
        """
        raise NotImplementedError()
    
    def run(self, lipids: LipidGroup) -> list[Leaflet]:
        """
        This method should return a list of leaflets.

        Parameters
        ----------
        lipids: LipidGroup
            LipidGroup containing lipid coordinates

        Returns
        -------
        leaflets: list[Leaflet]
            List of Leaflet objects
        """
        indices = self._run(lipids)
        return [
            Leaflet(lipids[cluster])
            for cluster in indices
        ]



class GraphMethod(GroupingMethod):

    def __init__(
        self,
        sparse: bool = False,
        cutoff: float = 12,
        n_leaflets: int = 2,
    ):
        self.sparse = sparse
        self.returntype = "numpy" if not sparse else "sparse"
        self.cutoff = cutoff
        self.n_leaflets = n_leaflets

    def _get_adjacency_matrix(self, lipids: LipidGroup) -> np.ndarray:
        """
        Get adjacency matrix from lipid coordinates.

        Parameters
        ----------
        lipids: LipidGroup
            LipidGroup containing lipid coordinates
        
        Returns
        -------
        adj: np.ndarray
            Adjacency matrix
        
        Raises
        ------
        ValueError
            If the NxN matrix is too big and `sparse` is not True
        """
        from MDAnalysis.analysis.distances import contact_matrix

        coordinates = self.lipids.unwrapped_headgroup_centers

        try:
            adj = contact_matrix(
                coordinates,
                cutoff=self.cutoff,
                box=lipids.universe.dimensions,
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
        return adj

    def _run(self, lipids: LipidGroup) -> list[Leaflet]:
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                'networkx is required to use this method '
                'but is not installed. Install it with '
                '`conda install networkx` or '
                '`pip install networkx`.'
            ) from None
        
        adj = self._get_adjacency_matrix(lipids)
        graph = nx.Graph(adj)
        groups = sorted(
            [list(c) for c in nx.connected_components(graph)],
            key=len,
            reverse=True,
        )[:self.n_leaflets]
        return groups

    


class SpectralClusteringMethod(GroupingMethod):
    def __init__(
        self,
        delta: typing.Optional[float] = 20,
        cosine_threshold: float = 0.8,
        angle_factor: float = 1,
        cutoff: float = 12,
        n_leaflets: int = 2,
    ):
        try:
            from sklearn.cluster import SpectralClustering
        except ImportError:
            raise ImportError(
                'scikit-learn is required to use this method '
                'but is not installed. Install it with '
                '`conda install scikit-learn` or '
                '`pip install scikit-learn`.'
            ) from None

        self.n_leaflets = n_leaflets
        self.delta = delta
        self.cosine_threshold = cosine_threshold
        self.angle_factor = angle_factor
        self.cutoff = cutoff
        self.predictor = SpectralClustering(
            n_clusters=self.n_leaflets,
            affinity="precomputed",
        )

    def _get_delta(self, distance_matrix):
        threshold = self.cutoff * 2
        return (
            self.delta
            or np.max(distance_matrix[distance_matrix < threshold]) / 3
        )
    
    def _normalize_angles_by_cosine_threshold(self, angles):
        cosine = np.clip(
            angles,
            -self.cosine_threshold,
            self.cosine_threshold,
        ) + self.cosine_threshold
        return cosine / (2 * self.cosine_threshold)

    def _get_kernel(self, lipids: LipidGroup) -> np.ndarray:
        from MDAnalysis.analysis.distances import self_distance_array
        from lipyds.lib.mdautils import get_distances_with_projection

        coordinates = lipids.unwrapped_headgroup_centers
        orientations = lipids.normalized_orientations
        
        distance_matrix = get_distances_with_projection(
            coordinates,
            orientations,
            self.cutoff,
            box=lipids.universe.dimensions,
            angle_factor=self.angle_factor,
        )

        delta = self._get_delta(distance_matrix)
        gaussian = np.exp(
            -distance_matrix ** 2
            / (2. * delta ** 2)
        )
        angles = np.dot(orientations, orientations.T)
        cosine = self._normalize_angles_by_cosine_threshold(angles)

        mask = ~np.isnan(cosine)
        gaussian[mask] *= cosine[mask]
        return gaussian
    
    def _run(self, lipids: LipidGroup) -> list[list[int]]:
        kernel = self._get_kernel(lipids)
        data_labels = self.predictor.fit_predict(kernel)
        ix = np.argsort(data_labels)
        local_indices = np.arange(len(lipids))
        where_to_split = np.where(np.ediff1d(data_labels[ix]))[0] + 1
        cluster_indices = np.split(local_indices[ix], where_to_split)
        return cluster_indices
