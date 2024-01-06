import typing

import pyvista as pv
import numpy as np

from .groups import Leaflet, Bilayer

AxisLabels = typing.Literal["x", "y", "z"]
LeafletLabels = typing.Literal["lower", "upper"]

class Surface:

    def __init__(
        self,
        surface: pv.PolyData,
        lipid_point_indices: np.ndarray,
        other_point_indices: np.ndarray,
        padded_point_indices: np.ndarray,
        all_point_normals: typing.Optional[np.ndarray] = None,
    ):
        self._surface = surface
        self._all_points = surface.points
        self._lipid_point_indices = lipid_point_indices
        self._other_point_indices = other_point_indices
        self._padded_point_indices = padded_point_indices
        self._all_point_normals = all_point_normals

    def _to_serializable(self):
        return {
            "surface": {
                "points": self._surface.points,
                "faces": self._surface.faces,
                "n_faces": self._surface.n_faces,
                "lines": self._surface.lines,
                "n_lines": self._surface.n_lines,
            },
            "lipid_point_indices": self._lipid_point_indices,
            "other_point_indices": self._other_point_indices,
            "padded_point_indices": self._padded_point_indices,
            "all_point_normals": self._all_point_normals,
        }
    
    def _to_npz_serializable(self, i: int, name: LeafletLabels = "lower"):
        normals = self.all_point_normals
        if normals is None:
            normals = np.nan

        return {
            f"points-{i}_{name}": self._surface.points,
            f"faces-{i}_{name}": self._surface.faces,
            f"n_faces-{i}_{name}": self._surface.n_faces,
            f"lines-{i}_{name}": self._surface.lines,
            f"n_lines-{i}_{name}": self._surface.n_lines,
            f"lipid_point_indices-{i}_{name}": self._lipid_point_indices,
            f"other_point_indices-{i}_{name}": self._other_point_indices,
            f"padded_point_indices-{i}_{name}": self._padded_point_indices,
            f"all_point_normals-{i}_{name}": normals
        }
    
    @classmethod
    def _from_npz_serializable(
        cls,
        serializable: dict[str, dict],
        i: int,
        name: LeafletLabels = "lower",
    ):
        surface_args = {}
        for kwarg in (
            "points",
            "faces",
            "n_faces",
            "lines",
            "n_lines",
        ):
            surface_args[kwarg] = serializable.get(f"{kwarg}-{i}_{name}")

        serializable_dict = {
            "surface": surface_args,
        }
        for kwarg in (
            "lipid_point_indices",
            "other_point_indices",
            "padded_point_indices",
            "all_point_normals",
        ):
            serializable_dict[kwarg] = serializable[f"{kwarg}-{i}_{name}"]
        
        if serializable_dict["all_point_normals"] is np.nan:
            serializable_dict["all_point_normals"] = None
        return cls._from_serializable(serializable_dict)
    

    
    @classmethod
    def _from_serializable(cls, serializable: dict[str, dict]):
        serializable = dict(serializable)
        surface_data = dict(serializable.pop("surface"))
        surface = pv.PolyData(
            surface_data.pop("points"),
            **surface_data
        )
        return cls(
            surface,
            **serializable,
        )
    
    def _to_serializable(self):
        return {
            "surface": {
                "points": self._surface.points,
                "faces": self._surface.faces,
                "n_faces": self._surface.n_faces,
                "lines": self._surface.lines,
                "n_lines": self._surface.n_lines,
            },
            "lipid_point_indices": self._lipid_point_indices,
            "other_point_indices": self._other_point_indices,
            "padded_point_indices": self._padded_point_indices,
            "all_point_normals": self._all_point_normals,
        }

    @classmethod
    def from_leaflet(
        cls,
        leaflet: Leaflet,
        select_other: str = "protein",
        cutoff_other: float = 6,
        pad_dimensions: list[AxisLabels] = ["x", "y"],
        padding_width: float = 8,
    ):

        all_points = leaflet.unwrapped_headgroup_centers
        lipid_point_indices = np.arange(len(all_points))
        n_points = len(all_points)

        # add other points
        if select_other is not None:
            other_atoms = leaflet.select_around_unwrapped_headgroup_centers(
                select_other, cutoff_other
            )
            other_points = other_atoms.positions
            all_points = np.vstack((all_points, other_points))

        other_point_indices = np.arange(n_points, len(all_points))
        n_points = len(all_points)

        # add padding around border to account for PBC
        if (
            leaflet.universe.dimensions is not None
            and padding_width
            and pad_dimensions
        ):
            # we can't handle triclinic boxes yet,
            # and is it even common to simulate membranes in triclinic?
            angles = leaflet.universe.dimensions[3:]
            box = leaflet.universe.dimensions[:3]
            if not np.all(angles == 90):
                raise NotImplementedError(
                    "Triclinic boxes are not supported for padding surfaces. "
                    "Please open an issue if you need this feature. "
                    "Set `padding_width=0` to disable padding. "
                    "Please note that this may lead to artifacts "
                    "at the box boundaries."
                )
            
            # add padding to all dimensions
            axes = {"x": 0, "y": 1, "z": 2}
            for axis_label, axis_index in axes.items():
                if axis_label in pad_dimensions:
                    axis = all_points[:, axis_index]
                    lower = axis < padding_width
                    lower_pad = axis[lower] + box[axis_index]

                    upper = axis > box[axis_index] - padding_width
                    upper_pad = axis[upper] - box[axis_index]

                    all_points = np.vstack((all_points, lower_pad, upper_pad))
        
        padded_point_indices = np.arange(n_points, len(all_points))
        surface = pv.PolyData(all_points).delaunay_2d()
        return cls(
            surface,
            lipid_point_indices,
            other_point_indices,
            padded_point_indices,
        )

    def compute_point_normals(
        self,
        cutoff: float = 10,
        reference_normal: list[float] = [0, 0, 1],
        box=None,
    ):
        from lipyds.lib.pyvutils import compute_surface_normals
        from MDAnalysis.lib.distances import self_capped_distance

        normals = compute_surface_normals(
            self._surface,
            global_normal=reference_normal,
        )
        if cutoff:
            # average point normals with neighbors for smoother surface
            pairs = self_capped_distance(
                self.lipid_points,
                cutoff,
                box=box,
                return_distances=False,
            )
            for lipid_index in range(self.n_lipid_points):
                neighbors = pairs[np.any(pairs == lipid_index, axis=1)]
                points = np.unique(neighbors)
                normals[lipid_index] = normals[points].mean(axis=0)
        
        self._all_point_normals = normals

    @property
    def all_point_normals(self):
        if self._all_point_normals is None:
            raise ValueError(
                "Surface normals have not been computed yet. "
                "Call `compute_point_normals` first."
            )
        return self._all_point_normals

    @property
    def leaflet(self):
        return self._leaflet

    @property
    def universe(self):
        return self.leaflet.universe
    
    @property
    def lipid_point_indices(self):
        return self._lipid_point_indices

    @property
    def n_lipid_points(self):
        return len(self.lipid_point_indices)
    
    @property
    def lipid_points(self):
        return self._all_points[self._lipid_point_indices]
    
    @property
    def lipid_point_normals(self):
        return self.all_point_normals[self._lipid_point_indices]

    @property
    def other_points(self):
        return self._all_points[self._other_point_indices]
    
    @property
    def padded_points(self):
        return self._all_points[self._padded_point_indices]
    


class SurfaceBilayer:

    @classmethod
    def from_bilayer(
        cls,
        bilayer: Bilayer,
        select_other: str = "protein",
        cutoff_other: float = 6,
        cutoff_normal_neighbors: float = 10,
        pad_dimensions: list[AxisLabels] = ["x", "y"],
        padding_width: float = 8,
        reference_normal: list[float] = [0, 0, 1],

    ):
        surfaces = []
        for leaflet in bilayer:
            surface = Surface.from_leaflet(
                leaflet,
                select_other=select_other,
                cutoff_other=cutoff_other,
                pad_dimensions=pad_dimensions,
                padding_width=padding_width,
            )
            surface.compute_point_normals(
                cutoff=cutoff_normal_neighbors,
                reference_normal=reference_normal,
                box=leaflet.universe.dimensions,
            )
            surfaces.append(surface)
        return cls(surfaces)
    

    def _to_serializable(self):
        return {
            "surfaces": [
                surface._to_serializable()
                for surface in self._surfaces
            ]
        }
    
    def _to_npz_serializable(self, i: int) -> dict[str, dict]:
        serializable = {}
        serializable.update(
            self.leaflets[0]._to_npz_serializable(i, name="lower")
        )
        serializable.update(
            self.leaflets[1]._to_npz_serializable(i, name="upper")
        )
        return serializable

    @classmethod
    def _from_serializable(cls, serializable: dict[str, dict]):
        serializable = dict(serializable)
        return cls(
            [
                Surface._from_serializable(surface)
                for surface in serializable.pop("surfaces")
            ]
        )
    
    @classmethod
    def _from_npz_serializable(cls, serializable: dict[str, dict], i: int):
        return cls(
            [
                Surface._from_npz_serializable(serializable, i, name="lower"),
                Surface._from_npz_serializable(serializable, i, name="upper"),
            ]
        )

    def __init__(self, surfaces: list[Surface]):
        self._surfaces = tuple(surfaces)


class SurfaceBilayerTrajectory:
    def __init__(self, surface_bilayers: list[SurfaceBilayer]):
        self._surface_bilayers = np.asarray(surface_bilayers)

    def __getitem__(self, index):
        return self._surface_bilayers[index]
    
    def __len__(self):
        return len(self._surface_bilayers)

    def _to_npz_serializable(self) -> dict[str, dict]:
        serializable = {}
        for i, surface_bilayer in enumerate(self._surface_bilayers):
            serializable.update(surface_bilayer._to_npz_serializable(i))
        return serializable
    
    def to_file(self, filename: str):
        np.savez_compressed(
            filename,
            **self._to_npz_serializable(),
        )
    
    @classmethod
    def from_file(cls, filename: str):
        serializable = np.load(filename, allow_pickle=False)
        return cls._from_npz_serializable(serializable)
    
    @classmethod
    def _from_npz_serializable(
        cls,
        serializable: dict[str, dict],
    ):
        point_keys = [
            key
            for key in serializable.keys()
            if key.startswith("points")
            and key.endswith("lower")
        ]
        frame_indices = np.unique(
            [
                int(key.split("-")[-1].split("_")[0])
                for key in point_keys
            ]
        )
        n_frames = max(frame_indices) + 1
        
        assert n_frames == len(frame_indices)
        # check that all keys are present
        # there are 9 keys per leaflet, 2 leaflets per bilayer
        n_expected = 9 * 2 * n_frames
        if len(serializable.keys()) != n_expected:
            raise ValueError(
                "The given serializable does not contain "
                "the expected number of keys. "
                f"Expected {n_expected} keys, "
                f"but {len(serializable.keys())} were found."
            )
        
        surface_bilayers = []
        for frame_index in range(n_frames):
            surface_bilayers.append(
                SurfaceBilayer._from_npz_serializable(
                    serializable,
                    frame_index,
                )
            )
        return cls(surface_bilayers)