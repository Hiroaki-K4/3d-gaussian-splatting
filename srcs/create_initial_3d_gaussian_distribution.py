from pathlib import Path

import numpy as np
import tyro
from scipy.spatial import cKDTree
from viser.extras.colmap import (
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary,
)


def create_initial_covariance(points):
    kdtree = cKDTree(points)
    k = 4  # 1 self-point + 3 nearest neighbors
    distances, indices = kdtree.query(points, k=k)

    radius = np.mean(distances[:, 1:], axis=1)
    covariances = [np.diag([r ** 2, r ** 2, r ** 2]) for r in radius]
    return covariances, radius


def main(
    colmap_path: Path = Path(__file__).parent / "../datasets/garden/sparse/0",
    images_path: Path = Path(__file__).parent / "../datasets/garden/images_8",
) -> None:
    # Load the colmap info.
    cameras = read_cameras_binary(colmap_path / "cameras.bin")
    images = read_images_binary(colmap_path / "images.bin")
    points3d = read_points3d_binary(colmap_path / "points3D.bin")
    points = np.array([points3d[p_id].xyz for p_id in points3d])
    colors = np.array([points3d[p_id].rgb for p_id in points3d])
    covariances, radius = create_initial_covariance(points)


if __name__ == "__main__":
    tyro.cli(main)
