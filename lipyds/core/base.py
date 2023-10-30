import numpy as np
from lipyds.lib.utils import cached_property

class ContainsLeafletMixin:
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
