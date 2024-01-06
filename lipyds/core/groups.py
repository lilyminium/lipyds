import typing
import MDAnalysis as mda
import numpy as np


from lipyds.lib.utils import cached_property

class Lipid:
    """
    This class represents a lipid molecule.

    Parameters
    ----------
    headgroup: mda.core.groups.AtomGroup or mda.core.groups.Atom
        The headgroup atom/s of the lipid.
    tailgroup: mda.core.groups.AtomGroup or mda.core.groups.Atom, optional
        The tailgroup atoms of the lipid.
        If ``None``, the tailgroup is every atom in the same residue
        as the headgroup that is not the headgroup.


    Attributes
    ----------
    atoms: mda.core.groups.AtomGroup
        The atoms of the lipid
    residue: mda.core.groups.Residue
        The lipid residue
    universe: mda.core.universe.Universe
        The universe the lipid belongs to
    headgroup: mda.core.groups.AtomGroup
        The headgroup atom/s of the lipid
    tailgroup: mda.core.groups.AtomGroup
        The tailgroup atoms of the lipid
    positions: np.ndarray
        The unprocessed positions of the lipid atoms
    """
    def __init__(
        self,
        headgroup: typing.Union[
            mda.core.groups.AtomGroup,
            mda.core.groups.Atom
        ],
        tailgroup: typing.Optional[
            typing.Union[
                mda.core.groups.AtomGroup,
                mda.core.groups.Atom
            ]
        ] = None,
    ):
        self._cache = {}
        self._unwrapped_residue_positions_frame = -1
        self._headgroup = sum([headgroup]) # make sure it's an AtomGroup
        if not self.headgroup.residues.n_residues == 1:
            raise ValueError(
                "headgroup atoms must belong to a single residue"
            )
        self._residue = self.headgroup.residues[0]

        if tailgroup is None:
            tailgroup = self.residue.atoms - self.headgroup
        self._tailgroup = sum([tailgroup])
        try:
            residue, = self.tailgroup.residues
            assert residue == self.residue
        except (ValueError, AssertionError):
            raise ValueError(
                "tailgroup atoms must belong to a same residue as headgroup"
            )
        
        self._atoms = self.headgroup + self.tailgroup
        
        #: dict[int, int]: mapping from absolute universe atom indices
        # to relative atom indices in the residue
        self._absolute_to_relative_indices = {
            atom.index: i for i, atom in enumerate(self.residue.atoms)
        }
        #: np.ndarray[int]: absolute universe atom indices of the headgroup
        self._absolute_headgroup_indices = self._headgroup.indices
        #: np.ndarray[int]: absolute universe atom indices of the tailgroup
        self._absolute_tailgroup_indices = self._tailgroup.indices

        #: np.ndarray[int]: relative residue atom indices of the headgroup
        self._relative_headgroup_indices = np.array([
            self._absolute_to_relative_indices[i]
            for i in self._absolute_headgroup_indices
        ])
        #: np.ndarray[int]: relative residue atom indices of the tailgroup
        self._relative_tailgroup_indices = np.array([
            self._absolute_to_relative_indices[i]
            for i in self._absolute_tailgroup_indices
        ])

        #: mda.core.groups.Atom: the first atom of the headgroup
        self._first_headgroup_atom = self._headgroup[0]
        self._leaflet = None

    def copy(self):
        return type(self)(self.headgroup, self.tailgroup)

    @property
    def leaflet(self):
        return self._leaflet

    @classmethod
    def from_atom_selections(
        cls,
        universe_or_atomgroup: typing.Union[
            mda.core.universe.Universe,
            mda.core.groups.AtomGroup
        ],
        select_headgroups: str = "all",
        select_tailgroups: typing.Optional[str] = None,
    ) -> list["Lipid"]:
        headgroup_atoms = universe_or_atomgroup.select_atoms(select_headgroups)
        index_to_headgroup = {
            ag.residues[0].resindex: ag
            for ag in headgroup_atoms.split("residue")
        }
        index_to_tailgroup = {}
        if select_tailgroups is not None:
            tail_atoms = universe_or_atomgroup.select_atoms(select_tailgroups)
            for ag in tail_atoms.split("residue"):
                index_to_tailgroup[ag.residues[0].resindex] = ag
        lipids = [
            cls(index_to_headgroup[i], index_to_tailgroup.get(i, None))
            for i in index_to_headgroup
        ]
        return lipids
        
    @property
    def headgroup(self) -> mda.core.groups.AtomGroup:
        return self._headgroup
    
    @property
    def tailgroup(self) -> mda.core.groups.AtomGroup:
        return self._tailgroup

    @property
    def residue(self) -> mda.core.groups.Residue:
        return self._residue
    
    @property
    def universe(self) -> mda.core.universe.Universe:
        return self.residue.universe
    
    @property
    def residue_positions(self) -> np.ndarray:
        return self.residue.atoms.positions

    @property
    def headgroup_positions(self) -> np.ndarray:
        """The absolute headgroup positions of the lipid"""
        return self._positions[self._relative_headgroup_indices]

    @property
    def tailgroup_positions(self) -> np.ndarray:
        """The absolute tailgroup positions of the lipid"""
        return self._positions[self._relative_tailgroup_indices]
    
    @property
    def unwrapped_residue_positions(self) -> np.ndarray:
        """The unwrapped positions of the lipid"""
        if self._unwrapped_residue_positions_frame != self.universe.trajectory.frame:
            self._cache.pop("_unwrapped_residue_positions", None)
        return self._unwrapped_residue_positions
    
    @cached_property
    def _unwrapped_residue_positions(self):
        unwrapped = self._get_unwrapped_residue_positions()
        self._unwrapped_residue_positions_frame = self.universe.trajectory.frame
        return unwrapped
    
    
    @property
    def unwrapped_headgroup_positions(self) -> np.ndarray:
        """The unwrapped headgroup positions of the lipid"""
        return self._unwrapped_residue_positions[self._relative_headgroup_indices]
    
    @property
    def unwrapped_tailgroup_positions(self) -> np.ndarray:
        """
        The unwrapped tailgroup positions of the lipid,
        unwrapped to the headgroup center.
        """
        return self._unwrapped_residue_positions[self._relative_tailgroup_indices]
    
    @property
    def unwrapped_headgroup_center(self):
        """The center of the headgroup atoms of the lipid"""
        if self.headgroup.n_atoms < 2:
            return self.unwrapped_headgroup_positions.reshape((3,))
        return np.mean(self.unwrapped_headgroup_positions, axis=0)
    
    @property
    def unwrapped_tailgroup_center(self) -> np.ndarray:
        """
        The center of the tailgroup atoms of the lipid,
        unwrapped to the headgroup center.
        """
        if self.tailgroup.n_atoms < 2:
            return self.unwrapped_tailgroup_positions.reshape((3,))
        return np.mean(self.unwrapped_tailgroup_positions, axis=0)
    
    @property
    def orientation(self) -> np.ndarray:
        """The orientation vector of the lipid
        
        This is calculated from the center of the head
        to the center of the tail.
        """
        return self.unwrapped_tailgroup_center - self.unwrapped_headgroup_center
    
    @property
    def normalized_orientation(self) -> np.ndarray:
        """The normalized orientation vector of the lipid"""
        orientation = self.orientation
        return orientation / np.linalg.norm(orientation)
    
    
    def _get_unwrapped_residue_positions(self):
        from lipyds.lib.mdautils import unwrap_coordinates

        return unwrap_coordinates(
            self.residue_positions,
            self._first_headgroup_atom.position,
            box=self.universe.dimensions,
        )



class LipidGroup:
    """
    This is a group of Lipids.

    Parameters
    ----------
    lipids: list[Lipid]
        The lipids in the group


    Attributes
    ----------
    universe: mda.core.universe.Universe
        The universe the lipids belong to
    lipids: np.ndarray[Lipid]
        The lipids in the group
    residues: mda.core.groups.ResidueGroup
        The residues of the lipids in the group
    n_residues: int
        The number of residues in the group
    unwrapped_headgroup_centers: np.ndarray
        The unwrapped headgroup centers of the lipids in the group
    """

    def __getitem__(self, index):
        return self.lipids[index]

    @classmethod
    def from_atom_selections(
        cls,
        universe_or_atomgroup: typing.Union[
            mda.core.universe.Universe,
            mda.core.groups.AtomGroup
        ],
        select_headgroups: str = "all",
        select_tailgroups: typing.Optional[str] = None,
    ) -> "LipidGroup":
        lipids = Lipid.from_atom_selections(
            universe_or_atomgroup,
            select_headgroups=select_headgroups,
            select_tailgroups=select_tailgroups,
        )
        return cls(lipids)


    def __init__(
        self,
        lipids: list[Lipid],
    ):
        self._cache = {}
        self._unwrapped_residue_positions_frame = -1
        self._lipids = np.array(lipids)

        self._residues = sum([
            lipid.residue for lipid in self._lipids
        ])
        self._universe = self._residues.universe

    def __iter__(self):
        return iter(self._lipids)

    @property
    def lipids(self) -> np.ndarray[Lipid]:
        """The lipids in the group"""
        return self._lipids
    
    @property
    def residues(self) -> mda.core.groups.ResidueGroup:
        """The residues of the lipids in the group"""
        return self._residues
    
    @property
    def universe(self) -> mda.core.universe.Universe:
        """The universe the lipids belong to"""
        return self._universe
    
    @property
    def n_residues(self) -> int:
        """The number of residues in the group"""
        return self.residues.n_residues
    
    def __len__(self):
        return len(self.lipids)
    
    def _check_unwrapped_frame(self):
        if self._unwrapped_residue_positions_frame != self.universe.trajectory.frame:
            self._cache.pop("_unwrapped_headgroup_centers", None)
            self._cache.pop("_normalized_orientations", None)
            self._unwrapped_residue_positions_frame = self.universe.trajectory.frame

    @property
    def unwrapped_headgroup_centers(self) -> np.ndarray:
        """The unwrapped headgroup center positions of the lipids in the group"""
        self._check_unwrapped_frame()
        return self._unwrapped_headgroup_centers
    
    @cached_property
    def _unwrapped_headgroup_centers(self):
        coordinates = np.array([
            lipid.unwrapped_headgroup_center
            for lipid in self.lipids
        ])
        return coordinates
    
    @property
    def normalized_orientations(self) -> np.ndarray:
        """The normalized orientations of the lipids in the group"""
        self._check_unwrapped_frame()
        return self._normalized_orientations
    
    @cached_property
    def _normalized_orientations(self):
        orientations = np.array([
            lipid.normalized_orientation
            for lipid in self.lipids
        ])
        return orientations
    
    def select_around_unwrapped_headgroup_centers(
        self,
        select: str = "all",
        cutoff: float = 6,
    ) -> mda.core.groups.AtomGroup:
        """
        Select atoms within distance ``cutoff``
        of the unwrapped headgroup centers of the lipids in the group.
        
        Parameters
        ----------
        select: str
            The selection string to select atoms
        cutoff: float
            The cutoff distance in Angstroms

        Returns
        -------
        mda.core.groups.AtomGroup
            The selected atoms within distance ``cutoff``
        """
        from MDAnalysis.lib.distances import capped_distance

        ag = self.universe.select_atoms(select)

        pairs = capped_distance(
            self.unwrapped_headgroup_centers,
            ag.positions,
            max_cutoff=cutoff,
            box=self.universe.dimensions,
            return_distances=False,
        )
        atom_indices = np.unique(pairs[:, 1])
        return ag[atom_indices]



class Leaflet(LipidGroup):
    def __init__(
        self, lipids: list[Lipid],
        leaflet_index: int = -1,
    ):
        lipids = sorted(
            lipids,
            key=lambda lipid: lipid.residue.resindex,
        )
        super().__init__(lipids)
        for lipid in self._lipids:
            lipid._leaflet = self
        self._leaflet_index = leaflet_index
        self._bilayer = None

class Bilayer:

    def __iter__(self):
        return iter(self._leaflets)

    def __init__(
        self,
        leaflets: list[Leaflet],
        sort_by: str = "z"
    ):
        self._cache = {}

        if sort_by == "z":
            z_avg = [
                leaflet.unwrapped_headgroup_centers[:, 2].mean()
                for leaflet in leaflets
            ]
            indices = np.argsort(z_avg)
        else:
            raise ValueError(
                f"Invalid sort_by: {sort_by}. "
                "Only 'z' is supported."
            )
        
        leaflets = tuple(leaflets[i] for i in indices)


        self._leaflets = tuple(leaflets)
        self._universe = self._leaflets[0].universe
        for i, leaflet in enumerate(self._leaflets):
            leaflet._bilayer = self
            leaflet._leaflet_index = i

