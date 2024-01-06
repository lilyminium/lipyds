import typing

import numpy as np
from MDAnalysis.analysis.base import AnalysisBase
from lipyds.core.groups import LipidGroup

from lipyds.leafletfinder.leafletfinder import BilayerFinder
from lipyds.leafletfinder.grouping import GroupingMethod


class AssignBilayerTrajectory(AnalysisBase):
    def __init__(
        self,
        universe,
        method: GroupingMethod,
        select_headgroups: str = "all",
        select_tailgroups: typing.Optional[str] = None,
    ):
        super().__init__(universe.trajectory)

        if not isinstance(method, GroupingMethod):
            raise TypeError(
                f"method must be an instance of GroupingMethod, "
                f"but {type(method)} was given."
            )
        if not method.n_leaflets == 2:
            raise ValueError(
                f"method must have n_leaflets=2, "
                f"but n_leaflets={method.n_leaflets} was given."
            )
        self.method = method
        self.lipids = LipidGroup.from_atom_selections(
            universe,
            select_headgroups=select_headgroups,
            select_tailgroups=select_tailgroups,
        )

    def _prepare(self):
        self.results.bilayers = []
    
    def _single_frame(self):
        bilayer = BilayerFinder(self.universe, self.method).run()
        self.results.bilayers.append(bilayer)
    
    def _conclude(self):
        self.results.bilayers = np.array(self.results.bilayers)