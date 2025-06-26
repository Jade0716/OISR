import numpy as np


width = 800
height = 800
fovx=np.deg2rad(35.0)
fovy=np.deg2rad(35.0)
cx = width / 2
cy = height / 2
fx = (width / 2) / np.tan(fovx / 2)
fy = (height / 2) / np.tan(fovy / 2)

K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1]
])

print("Camera intrinsic matrix:\n", K)
