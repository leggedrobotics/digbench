import os
import numpy as np
import cv2

package_dir = os.path.dirname(os.path.abspath(__file__))
path = package_dir + '/../data/openstreet/train/benchmark_60_60/terra/small-rectangles/images/img_3.npy'
m = np.load(path)
print(m.shape)
print(m)
mc = m.astype(np.uint8) * 255
cv2.imshow("m", mc.repeat(5, 0).repeat(5, 1))
cv2.waitKey(0)