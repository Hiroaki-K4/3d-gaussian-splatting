import random
import time
from pathlib import Path
from typing import List

import imageio.v3 as iio
import numpy as np
import tyro
import viser
import viser.transforms as tf
from tqdm.auto import tqdm
from viser.extras.colmap import (
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary,
)

from create_initial_3d_gaussian_distribution import create_initial_covariance


def main(
    colmap_path: Path = Path(__file__).parent / "../datasets/garden/sparse/0",
    images_path: Path = Path(__file__).parent / "../datasets/garden/images_8",
    downsample_factor: int = 2,
) -> None:
    """Visualize COLMAP sparse reconstruction outputs.

    Args:
        colmap_path: Path to the COLMAP reconstruction directory.
        images_path: Path to the COLMAP images directory.
        downsample_factor: Downsample factor for the images.
    """
    server = viser.ViserServer()
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Load the colmap info.
    cameras = read_cameras_binary(colmap_path / "cameras.bin")
    images = read_images_binary(colmap_path / "images.bin")
    points3d = read_points3d_binary(colmap_path / "points3D.bin")

    gui_points = server.gui.add_slider(
        "Max points",
        min=1,
        max=len(points3d),
        step=1,
        initial_value=min(len(points3d), 50_000),
    )
    gui_frames = server.gui.add_slider(
        "Max frames",
        min=1,
        max=len(images),
        step=1,
        initial_value=min(len(images), 100),
    )
    gui_point_size = server.gui.add_slider(
        "Point size", min=0.01, max=0.1, step=0.001, initial_value=0.05
    )

    points = np.array([points3d[p_id].xyz for p_id in points3d])
    colors = np.array([points3d[p_id].rgb for p_id in points3d])

    point_mask = np.random.choice(points.shape[0], gui_points.value, replace=False)
    covariances, radius = create_initial_covariance(points)
    point_sizes = np.round(radius, 3)
    for size in np.unique(point_sizes):
        indices = np.where(point_sizes == size)[0]
        name = f"/points_size_{size}"
        server.scene.add_point_cloud(
            name,
            points=points[indices],
            colors=colors[indices],
            point_size=size,
            point_shape="circle",  # sparkle is funny, try it
        )

    frames: List[viser.FrameHandle] = []
    draw_initial_3d_gaussians = server.gui.add_checkbox(
        "Draw initial 3D Gaussians", False
    )

    def visualize_frames() -> None:
        """Send all COLMAP elements to viser for visualization. This could be optimized
        a ton!"""

        # Remove existing image frames.
        for frame in frames:
            frame.remove()
        frames.clear()

        # Interpret the images and cameras.
        img_ids = [im.id for im in images.values()]
        random.shuffle(img_ids)
        img_ids = sorted(img_ids[: gui_frames.value])

        def attach_callback(
            frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle
        ) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        for img_id in tqdm(img_ids):
            img = images[img_id]
            cam = cameras[img.camera_id]

            # Skip images that don't exist.
            image_filename = images_path / img.name
            if not image_filename.exists():
                continue

            T_world_camera = tf.SE3.from_rotation_and_translation(
                tf.SO3(img.qvec), img.tvec
            ).inverse()
            frame = server.scene.add_frame(
                f"/colmap/frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.1,
                axes_radius=0.005,
            )
            frames.append(frame)

            # For pinhole cameras, cam.params will be (fx, fy, cx, cy).
            if cam.model != "PINHOLE":
                print(f"Expected pinhole camera, but got {cam.model}")

            H, W = cam.height, cam.width
            fy = cam.params[1]
            image = iio.imread(image_filename)
            image = image[::downsample_factor, ::downsample_factor]
            frustum = server.scene.add_camera_frustum(
                f"/colmap/frame_{img_id}/frustum",
                fov=2 * np.arctan2(H / 2, fy),
                aspect=W / H,
                scale=0.15,
                image=image,
            )
            attach_callback(frustum, frame)

    need_update = True

    @gui_points.on_update
    def _(_) -> None:
        point_mask = np.random.choice(points.shape[0], gui_points.value, replace=False)
        point_cloud.points = points[point_mask]
        point_cloud.colors = colors[point_mask]

    @gui_frames.on_update
    def _(_) -> None:
        nonlocal need_update
        need_update = True

    @gui_point_size.on_update
    def _(_) -> None:
        point_cloud.point_size = gui_point_size.value

    # @draw_initial_3d_gaussians.on_update
    # def _(_) -> None:
    #     if draw_initial_3d_gaussians.value:
    #         point_sizes = np.random.choice([0.01, 0.005, 0.02], size=points.shape[0])
    #         for size in np.unique(point_sizes):
    #             indices = np.where(point_sizes == size)[0]
    #             name = f"/points_size_{size}"
    #             server.scene.add_point_cloud(
    #                 name,
    #                 points=points[indices],
    #                 colors=colors[indices],
    #                 point_size=size,
    #                 point_shape="circle",  # sparkle is funny, try it
    #             )
    #     else:
    #         for size in np.unique(point_sizes):
    #             indices = np.where(point_sizes == size)[0]
    #             name = f"/points_size_{size}"
    #             server.scene.add_point_cloud(
    #                 name,
    #                 points=points[indices],
    #                 colors=colors[indices],
    #                 point_size=0.03,
    #                 point_shape="circle",  # sparkle is funny, try it
    #             )

    while True:
        if need_update:
            need_update = False
            visualize_frames()

        time.sleep(1e-3)


if __name__ == "__main__":
    tyro.cli(main)
