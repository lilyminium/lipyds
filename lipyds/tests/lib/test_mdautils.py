import pytest

import numpy as np

from lipyds.lib.mdautils import unwrap_coordinates

class TestUnwrap:

    @pytest.mark.parametrize(
        "center, unwrapped",
        [
            (
                [0, 0, 0],
                [
                    [0, -2, 0],
                    [1, -2, 0],
                    [-1, -2, 0],
                    [0, 2, 0],
                    [-1, 2, 0],
                ]
            ),
            (
                [0, 3, 0],
                [
                    [0, 3, 0],
                    [1, 3, 0],
                    [-1, 3, 0],
                    [0, 2, 0],
                    [-1, 2, 0],
                ]
            ),
            (
                [9, 2, 0],
                [
                    [10, 3, 0],
                    [11, 3, 0],
                    [9, 3, 0],
                    [10, 2, 0],
                    [9, 2, 0],
                ]
            )
        ]
    )
    def test_unwrap_small_ortho(self, center, unwrapped):
        """

            ------------------
            |                |
            |                |
            |XX             X|
            |X              X|
            |                |
            ------------------

        """
        coordinates = np.array([
            [0, 3, 0],
            [1, 3, 0],
            [9, 3, 0],
            [0, 2, 0],
            [9, 2, 0],
        ], dtype=float)

        box = np.array([10, 5, 3, 90, 90, 90], dtype=float)
        output = unwrap_coordinates(
            coordinates,
            center=center,
            box=box,
        )
        assert np.allclose(output, unwrapped)
