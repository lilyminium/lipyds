import typing

import MDAnalysis as mda

from lipyds.core.groups import LipidGroup, Leaflet, Bilayer
from lipyds.leafletfinder.grouping import GroupingMethod
class LeafletFinder:

    def __init__(
        self,
        universe,
        method: GroupingMethod,
        select_headgroups: str = "all",
        select_tailgroups: typing.Optional[str] = None,
    ):
        self._universe = universe
        self._lipids = LipidGroup.from_atom_selections(
            universe,
            select_headgroups=select_headgroups,
            select_tailgroups=select_tailgroups,
        )
        self.method = method
    
    def run(self):
        return self.method.run(self._lipids)


class BilayerFinder(LeafletFinder):
    def run(self):
        leaflets = super().run()
        assert len(leaflets) == 2, (
            "BilayerFinder requires n_leaflets=2, "
            f"but {len(leaflets)} were found."
        )
        bilayer = Bilayer(leaflets)
        return bilayer
