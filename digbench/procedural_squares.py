import numpy as np
import os
from pathlib import Path


def generate_squares(n_imgs, x_dim, y_dim, side_len, save_folder=None):
    """
    n_imgs, img_edge_min, img_edge_max, resolution=0.1, option=1, save_folder=save_folder
    option 1: visualize
    option 2: save to disk
    """
    side_len_contour = 3*side_len
    margin = 5
    print(f"{margin=}")
    for i in range(1, n_imgs+1):
        img = np.zeros((x_dim, y_dim))
        x = np.random.randint(0, x_dim - side_len_contour - 1, ())
        y = np.random.randint(0, y_dim - side_len_contour - 1, ())

        edge = np.random.randint(0, 4)
        if edge == 0:
            img[x + side_len_contour // 2:x+side_len_contour, y:y+side_len_contour] = 1
        elif edge == 1:
            img[x:x+side_len_contour // 2, y:y+side_len_contour] = 1
        elif edge == 2:
            img[x:x+side_len_contour, y + side_len_contour // 2:y+side_len_contour] = 1
        elif edge == 3:
            img[x:x+side_len_contour, y:y+side_len_contour // 2] = 1
        else:
            raise(RuntimeError(f"{edge=}"))


        x_margin = x+((side_len_contour-side_len) // 2)-margin
        y_margin = y+((side_len_contour-side_len) // 2)-margin
        img[x_margin:x_margin+side_len+2*margin, y_margin:y_margin+side_len+2*margin] = 0

        x_dig = x+((side_len_contour-side_len) // 2)
        y_dig = y+((side_len_contour-side_len) // 2)
        img[x_dig:x_dig+side_len, y_dig:y_dig+side_len] = -1

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
    side_len = 10
    package_dir = os.path.dirname(os.path.abspath(__file__))
    save_folder = package_dir + f'/../data/openstreet/train/squares_{side_len}/terra'
    generate_squares(n_imgs, x_dim, y_dim, side_len, save_folder=save_folder)
