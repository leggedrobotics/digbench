import numpy as np
import cv2
import os
import json
from pathlib import Path
from digbench.utils import color_dict
from digbench.utils import _get_img_mask

def generate_trenches(n_imgs, img_edge_min, img_edge_max, sizes_small, sizes_long, n_edges, min_trench_area_ratio, resolution, option, save_folder=None):
    """
    option 1: visualize
    option 2: save to disk
    """
    min_ssmall, max_ssmall = sizes_small
    min_slong, max_slong = sizes_long
    min_edges, max_edges = n_edges
    
    i = 0
    while i < n_imgs:
        w, h = np.random.randint(img_edge_min, img_edge_max, (2,), dtype=np.int32)
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
        
        # Set resolution
        img = cv2.resize(img, (int(img.shape[0] // resolution), int(img.shape[1] // resolution)), interpolation=cv2.INTER_NEAREST)

        img = img.astype(np.uint8)
        i += 1

        if option == 1:
            cv2.imshow("img", img)
            cv2.waitKey(0)
        elif option == 2:
            save_folder_images = Path(save_folder) / "images"
            save_folder_metadata = Path(save_folder) / "metadata"
            save_folder_images.mkdir(parents=True, exist_ok=True)
            save_folder_metadata.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(os.path.join(save_folder_images, "trench_" + str(i) + ".png"), img)
            with open(os.path.join(save_folder_metadata, "trench_" + str(i) + '.json'), 'w') as outfile:
                        json.dump({"real_dimensions": {"width": float(w), "height": float(h)}}, outfile)
        else:
            raise ValueError(f"Option {option} not supported.")

if __name__ == "__main__":
    n_imgs = 10
    img_edge_min, img_edge_max = 300, 600
    sizes_small = (15, 50)
    sizes_long = (100, 200)
    n_edges = (2, 2)
    min_trench_area_ratio = 0.02
    package_dir = os.path.dirname(os.path.abspath(__file__))
    save_folder = package_dir + '/../data/openstreet/benchmark_' + str(img_edge_min) + '_' + str(img_edge_max)
    generate_trenches(n_imgs, img_edge_min, img_edge_max, sizes_small, sizes_long, n_edges, min_trench_area_ratio, option=1, save_folder=save_folder)
