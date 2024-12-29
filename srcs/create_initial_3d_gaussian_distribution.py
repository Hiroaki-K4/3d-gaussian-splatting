from pathlib import Path

import numpy as np
import tyro
from tqdm import tqdm
from viser.extras.colmap import (
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary,
)


def create_initial_covariance(points):
    for i in tqdm(range(points.shape[0])):
        for j in range(points.shape[0]):
            continue


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
    create_initial_covariance(points)


if __name__ == "__main__":
    tyro.cli(main)
