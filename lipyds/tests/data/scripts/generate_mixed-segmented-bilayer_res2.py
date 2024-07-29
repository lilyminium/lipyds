import numpy as np
import MDAnalysis as mda

def generate_grid(
    start: int = 0,
    stop: int = 18,
    step: int = 10,
):
    xs = np.linspace(start, stop, step)
    ys = np.linspace(start, stop, step)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    xy = np.array([xx, yy]).T
    z = np.zeros((step, step, 1), dtype=float)
    xyz = np.concatenate([xy, z], axis=-1)
    return xyz

def generate_bilayer():
    # generate positions
    upper_positions = generate_grid().reshape((-1, 3))
    upper_positions[:, -1] = 60
    lower_positions = generate_grid().reshape((-1, 3))
    lower_positions[:, -1] = 20
    frame_1_positions = np.concatenate([upper_positions, lower_positions])
    frame_1_positions.shape

    frame_2_positions = np.concatenate([lower_positions, upper_positions])
    all_positions = np.array([frame_1_positions, frame_2_positions])

    # create universe
    u = mda.Universe.empty(
        n_atoms=200,
        n_residues=200,
        n_segments=1,
        n_frames=2,
        atom_resindex=np.arange(200),
        trajectory=True,
    )
    u.load_new(all_positions)

    u.dimensions = np.array([20, 20, 80, 90, 90, 90])

    # annotating with residue names
    row_1_names = ["POPC", "POPE"] * 5
    row_2_names = ["POPE", "POPC"] * 5
    upper_names = np.array([row_1_names, row_2_names] * 5).flatten()
    upper_names

    lower_names = np.array(["POPC"] * 50 + ["POPE"] * 50)
    all_names = np.concatenate([upper_names, lower_names])

    u.add_TopologyAttr("resnames", all_names)
    u.add_TopologyAttr("resids", np.arange(200))

    u.add_TopologyAttr("names")
    u.atoms.names = "PO4"

    with mda.Writer("../mixed-segmented-bilayer_res2.pdb") as writer:
        for ts in u.trajectory:
            writer.write(u.atoms)

if __name__ == "__main__":
    generate_bilayer()