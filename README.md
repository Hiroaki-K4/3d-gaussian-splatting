# 3d-gaussian-splatting

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

<br></br>

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
