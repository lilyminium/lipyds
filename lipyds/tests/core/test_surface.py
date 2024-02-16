import MDAnalysis as mda

import pytest

from lipyds.core.groups import Leaflet
from lipyds.core.surface import Surface, SurfaceBilayer
from lipyds.leafletfinder.leafletfinder import BilayerFinder
from lipyds.leafletfinder.grouping import GraphMethod
from lipyds.tests.datafiles import (
    NEURONAL_DDAT_GRO, NEURONAL_DDAT_XTC,
    NEURONAL_HSERT_UPPER_LAYER_PDB
)


class TestSurface:

    @pytest.fixture()
    def hsert_neuronal_upper_surface(self):
        u = mda.Universe(NEURONAL_HSERT_UPPER_LAYER_PDB)
        atomgroup = u.select_atoms("not protein")
        leaflet = Leaflet.from_atom_selections(
            atomgroup,
            select_headgroups="name PO4 GL* AM*"
        )
        assert len(leaflet) == 334
        surface = Surface.from_leaflet(leaflet)
        return surface

    def test_creation_from_leaflet(self, hsert_neuronal_upper_surface):
        assert hsert_neuronal_upper_surface.n_lipid_points == 334
        assert hsert_neuronal_upper_surface.n_all_points == 442


class TestSurfaceBilayer:

    @pytest.fixture()
    def ddat_neuronal_bilayer(self):
        u = mda.Universe(NEURONAL_DDAT_GRO)
        graph = GraphMethod(
            sparse=False,
            cutoff=10,
            n_leaflets=2
        )

        finder = BilayerFinder(
            u,
            method=graph,
            select_headgroups="name PO4 GL* AM* ROH"
        )
        bilayer = finder.run()
        
        surface_bilayer = SurfaceBilayer.from_bilayer(bilayer, select_other="protein")
        return surface_bilayer

    def test_surface_bilayer_creation(self, ddat_neuronal_bilayer):
        surface_bilayer = ddat_neuronal_bilayer
        assert len(surface_bilayer.surfaces) == 2
        lower = surface_bilayer.surfaces[0]
        upper = surface_bilayer.surfaces[1]


        assert lower.n_all_points == 918
        assert upper.n_all_points == 992

        assert lower.n_lipid_points == 564
        assert upper.n_lipid_points == 613

        assert lower.n_other_points == 133
        assert upper.n_other_points == 132

        assert lower.n_padded_points == 221
        assert upper.n_padded_points == 247
