import importlib_resources

__all__ = [
    "Martini_double_membrane",  # for testing leaflet finder more
    "DPPC_vesicle_only", # leaflet finder on vesicle
    "DPPC_vesicle_plus", # leaflet finder on vesicle
    "NEURONAL_GLYT2",  # neuronal membrane, DOI 10.1016/j.bbadva.2021.100010
    "NEURONAL_DDAT",  # neuronal membrane, DOI 10.1016/j.bbadva.2021.100010
    "NEURONAL_HDAT",  # neuronal membrane, DOI 10.1016/j.bbadva.2021.100010
    "NEURONAL_HSERT",  # neuronal membrane, DOI 10.1016/j.bbadva.2021.100010
    "DDAT_POPC_TPR",
    "DDAT_POPC_XTC",
    "SURFACE_POINTS",

    "SINGLE_POPC_UNWRAPPED",
    "SINGLE_POPC_WRAPPED",
]


data_directory = importlib_resources.files("lipyds") / "tests" / "data"

Martini_double_membrane = data_directory / "martini_double_bilayer.gro"
DPPC_vesicle_only = data_directory / "fatslim_dppc_vesicle.pdb"
DPPC_vesicle_plus = data_directory / "fatslim_dppc_vesicle_plus.gro"

NEURONAL_GLYT2 = data_directory / "glyt2_neuronal.gro"
NEURONAL_DDAT = data_directory / "ddat_neuronal.gro"
NEURONAL_HDAT = data_directory / "hdat_neuronal.gro"
NEURONAL_HSERT = data_directory / "hsert_neuronal.gro"

DDAT_POPC_TPR = data_directory / "dDAT_POPC-CHOL_r1_nowater.tpr"
DDAT_POPC_XTC = data_directory / "dDAT_POPC-CHOL_r1_10ns.xtc"
SURFACE_POINTS = data_directory / "surface_points.dat"

SINGLE_POPC_UNWRAPPED = data_directory / "single_POPC_unwrapped.pdb"
SINGLE_POPC_WRAPPED = data_directory / "single_POPC_wrapped.pdb"
