import typing
import MDAnalysis as mda

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
    residue: mda.core.groups.Residue
        The lipid residue
    headgroup: mda.core.groups.AtomGroup
        The headgroup atom/s of the lipid
    tailgroup: mda.core.groups.AtomGroup
        The tailgroup atoms of the lipid
    
    """
    @property
    def headgroup(self):
        return self._headgroup
    
    @property
    def tailgroup(self):
        return self._tailgroup

    @property
    def residue(self):
        return self._residue
    
    @property
    def universe(self):
        return self.residue.universe
        
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
            assert residue is self.residue
        except (ValueError, AssertionError):
            raise ValueError(
                "tailgroup atoms must belong to a same residue as headgroup"
            )
        




class Leaflet:
    """
    
    """

    def __init__(self, universe_or_atomgroup):
        self.universe = universe_or_atomgroup.universe
        self.atoms = universe_or_atomgroup.atoms
        self.residues = universe_or_atomgroup.residues