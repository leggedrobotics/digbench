import os
import numpy as np
import cv2

i = np.random.randint(1, 600)

package_dir = os.path.dirname(os.path.abspath(__file__))
# path = package_dir + f'/../data/openstreet/train/benchmark_20_40/terra/foundations/hard/images/img_{i}.npy'
# path = package_dir + f'/../data/openstreet/train/benchmark_20_40/terra/foundations/hard/occupancy/img_{i}.npy'
# path = package_dir + f'/../data/openstreet/train/benchmark_60_60/terra/rectangles_60/images/img_{i}.npy'
# path = package_dir + f'/../data/openstreet/train/benchmark_60_61/terra/trenches/easy/images/img_{i}.npy'

# path = package_dir + f'/../data/openstreet/train/benchmark_60_61/terra/trenches/easy/images/img_{i}.npy'
path = package_dir + f'/../data/openstreet/train/benchmark_20_50/terra/foundations/images/img_{i}.npy'

m = np.load(path)
print(m.shape)
print(m.max())
print(m.min())
print(m)
print(set(m.reshape(-1).tolist()))
m = (m+1)/2
mc = (m * 255).astype(np.uint8)
print(set(mc.reshape(-1).tolist()))

# mc = mc.repeat(5, 0).repeat(5, 1)
cv2.imshow("m", mc)
cv2.waitKey(0)