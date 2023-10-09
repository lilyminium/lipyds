from typing import Union
import warnings

import MDAnalysis as mda

def guess_lipid_headgroup(
    atomgroup: Union[mda.Universe, mda.core.groups.AtomGroup],
):
    """
    Guess the lipid headgroup for each residue by their residue name.
    The headgroups are inserted into the `lipid_headgroups` attribute
    inplace.

    If the headgroup cannot be guessed, a warning is raised.
    The guessing is based on the mapping in :mod:`lipyds.data.lipid_classes`.

    .. note::

        This function is likely to change when MDAnalysis incorporates
        context-specific guessers.

    Parameters
    ----------
    atomgroup : mda.Universe or mda.core.groups.AtomGroup
        :class:`~MDAnalysis.core.universe.Universe` or
        :class:`~MDAnalysis.core.groups.AtomGroup` to operate on.
    """
    from lipyds.data.lipid_classes import _HEADGROUP
    from lipyds.core.topologyattrs import LipidHeadgroup
    rg = atomgroup.residues
    rg.universe.add_TopologyAttr("lipid_headgroups")

    non_matches = set()
    for residue in rg:
        if residue.resname in _HEADGROUP:
            residue.lipid_headgroup = _HEADGROUP[residue.resname]
        else:
            non_matches.add(residue.resname)

    if non_matches:
        warnings.warn(
            f"Could not guess headgroup for {len(non_matches)} residues: "
            f"{non_matches}"
        )


def guess_lipid_tail_saturation(
    atomgroup: Union[mda.Universe, mda.core.groups.AtomGroup],
):
    """
    Guess the lipid tail saturation for each residue by their residue name.
    The tail saturations are inserted into the `lipid_tail_saturations`
    attribute inplace.

    If the tail saturation cannot be guessed, a warning is raised.
    The guessing is based on the mapping in :mod:`lipyds.data.lipid_classes`.

    .. note::

        This function is likely to change when MDAnalysis incorporates
        context-specific guessers.


    Parameters
    ----------
    atomgroup : mda.Universe or mda.core.groups.AtomGroup
        :class:`~MDAnalysis.core.universe.Universe` or
        :class:`~MDAnalysis.core.groups.AtomGroup` to operate on.
    """
    from lipyds.data.lipid_classes import _SATURATION
    from lipyds.core.topologyattrs import LipidTailSaturation
    rg = atomgroup.residues
    rg.universe.add_TopologyAttr("lipid_tail_saturations")

    non_matches = set()
    for residue in rg:
        if residue.resname in _SATURATION:
            residue.lipid_tail_saturation = _SATURATION[residue.resname]
        else:
            non_matches.add(residue.resname)

    if non_matches:
        warnings.warn(
            f"Could not guess tail saturation for {len(non_matches)} residues: "
            f"{non_matches}"
        )