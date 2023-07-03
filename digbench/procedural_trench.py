import numpy as np
import cv2
import os
from pathlib import Path
from digbench.utils import color_dict
from digbench.utils import _get_img_mask

def generate_trenches(n_imgs, w, h, sizes_small, sizes_long, n_edges, min_trench_area_ratio, option, save_folder=None):
    """
    option 1: visualize
    option 2: save to disk
    """
    min_ssmall, max_ssmall = sizes_small
    min_slong, max_slong = sizes_long
    min_edges, max_edges = n_edges
    
    i = 0
    while i < n_imgs:
        img = np.ones((w, h, 3)) * np.array(color_dict["neutral"])
        n_edges = np.random.randint(min_edges, max_edges + 1)

        prev_horizontal = True if np.random.choice([0,1]).astype(np.bool_) else False

        for edge_i in range(n_edges):
            if edge_i == 0:
                mask = np.ones_like(img[..., 0], dtype=np.bool_)
            fmask = mask.reshape(-1)
            fmask_idx_set = list(set((np.arange(w*h) * fmask).tolist()))[1:]  # remove idx 0 as it's always going to be present
            fidxs = np.array(fmask_idx_set)
            idx = np.random.choice(fidxs)
            x = idx // h
            y = idx % h
            size_small = np.random.randint(min_ssmall, max_ssmall + 1)
            size_long = np.random.randint(min_slong, max_slong + 1)
            if prev_horizontal:
                size_x = size_long
                size_y = size_small
            else:
                size_x = size_small
                size_y = size_long
            prev_horizontal = not prev_horizontal
            
            size_x = min(size_x, w-x)
            size_y = min(size_y, h-y)
            
            img[x:x+size_x, y:y+size_y] = np.array(color_dict["digging"])

            mask = np.zeros_like(img[..., 0], dtype=np.bool_)
            mask[x:x+size_x, y:y+size_y] = np.ones((size_x, size_y), dtype=np.bool_)

        if _get_img_mask(img, color_dict["digging"]).sum() / (w*h) < min_trench_area_ratio:
            print("skipping...")
            continue

        img = img.astype(np.uint8)
        i += 1

        if option == 1:
            cv2.imshow("img", img)
            cv2.waitKey(0)
        elif option == 2:
            Path(save_folder).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(os.path.join(save_folder, "trench_" + str(i) + ".png"), img)
        else:
            raise ValueError(f"Option {option} not supported.")

if __name__ == "__main__":
    n_imgs = 10
    w, h = 300, 600
    sizes_small = (15, 50)
    sizes_long = (100, 200)
    n_edges = (2, 2)
    min_trench_area_ratio = 0.02
    package_dir = os.path.dirname(os.path.abspath(__file__))
    save_folder = package_dir + '/../data/openstreet/benchmark_' + str(w) + '_' + str(h)
    generate_trenches(n_imgs, w, h, sizes_small, sizes_long, n_edges, min_trench_area_ratio, option=1, save_folder=save_folder)
