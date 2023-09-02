import numpy as np
import os
from pathlib import Path


def generate_squares(n_imgs, x_dim, y_dim, side_len, save_folder=None):
    """
    n_imgs, img_edge_min, img_edge_max, resolution=0.1, option=1, save_folder=save_folder
    option 1: visualize
    option 2: save to disk
    """

    for i in range(1, n_imgs+1):
        img = np.ones((x_dim, y_dim))
        x = np.random.randint(0, x_dim - side_len - 1, ())
        y = np.random.randint(0, y_dim - side_len - 1, ())
        img[x:x+side_len, y:y+side_len] = -1
        img = img.astype(np.int8)

        occ = np.zeros_like(img)

        save_folder_images = Path(save_folder) / "images"
        save_folder_occupancy = Path(save_folder) / "occupancy"
        save_folder_images.mkdir(parents=True, exist_ok=True)
        save_folder_occupancy.mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(save_folder_images, "img_" + str(i)), img)
        np.save(os.path.join(save_folder_occupancy, "img_" + str(i)), occ)
        print(f"Generated squares {i}")


if __name__ == "__main__":
    n_imgs = 1000
    x_dim = 60
    y_dim = 60
    side_len = 2
    package_dir = os.path.dirname(os.path.abspath(__file__))
    save_folder = package_dir + f'/../data/openstreet/train/squares_{side_len}/terra'
    generate_squares(n_imgs, x_dim, y_dim, side_len, save_folder=save_folder)
