# 3d-gaussian-splatting

<br></br>

# How to create a local environment

```bash
conda create -n 3dgs python=3.11
conda activate 3dgs
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

<br></br>

# Initialization of 3D Gaussians
First, we use COLMAP to obtain the initial point cloud and camera matrix.
Then, we create the initial 3D gaussians. The following elements are initialized.

- Positions
- Covariances
- Colors
- Opacities

The initial point cloud position is used directly as the initial gaussian position. 
The method for calculating initial covariance is explained in [the paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_low.pdf) as follows.

```
We estimate the initial covariance matrix as an isotropic Gaussian with axes equal to the mean ofthe distance to the closest three points.
```

I used nearest neighbor search to get closet three points. cKDTree of scipy is useful to do that. The code is as follows.

```python
def create_initial_covariance(points):
    kdtree = cKDTree(points)
    k = 4  # 1 self-point + 3 nearest neighbors
    distances, indices = kdtree.query(points, k=k)

    radius = np.mean(distances[:, 1:], axis=1)
    covariances = [np.diag([r ** 2, r ** 2, r ** 2]) for r in radius]
    return covariances, radius
```

[Visor](https://github.com/nerfstudio-project/viser) is very useful to visualize 3D information. You can visualize initial 3D Gaussians by running following commands.

```bash
cd srcs
python3 visualize_initial_3d_gaussians.py
```

<img src="resources/initial_gaussians.gif" width='600'>

It can be seen that where there are more point clouds, the radius is smaller, and where there are fewer point clouds, the radius is larger.

Since the visualization above visualizes position and radius, opacities is irrelevant, but is initialized at $0.1$. For color, the spherical harmonics coefficients are calculated from the RGB of the point cloud.

<br></br>

# References

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [Visor](https://github.com/nerfstudio-project/viser)
