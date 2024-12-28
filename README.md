# 3d-gaussian-splatting

```bash
conda create -n 3dgs python=3.11
conda activate 3dgs
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

```
We estimate the initial covariance matrix as an isotropic Gaussian with axes equal to the mean ofthe distance to the closest three points.
```
