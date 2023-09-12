import numpy as np
import os
from pathlib import Path


def generate_squares(n_imgs, x_dim, y_dim, side_len, save_folder=None):
    """
    The maps include:
    - dig area -> square
    - must dump area -> another square
    - can't dump area -> margin around the square
    - can't traverse area -> 3 small squares

    n_imgs, img_edge_min, img_edge_max, resolution=0.1, option=1, save_folder=save_folder
    """
    side_len_contour = 3*side_len
    margin = 5
    print(f"{margin=}")
    for i in range(1, n_imgs+1):
        img = np.zeros((x_dim, y_dim))
        occ = np.zeros_like(img)
        dmp = np.ones_like(img)

        x = np.random.randint(0, x_dim - side_len_contour - 1, ())
        y = np.random.randint(0, y_dim - side_len_contour - 1, ())

        # x_margin = x+((side_len_contour-side_len) // 2)-margin
        # y_margin = y+((side_len_contour-side_len) // 2)-margin
        # img[x_margin:x_margin+side_len+2*margin, y_margin:y_margin+side_len+2*margin] = 0

        img[x:x+side_len_contour, y:y+side_len_contour] = 1  # must dump
        x_dig = x+((side_len_contour-side_len) // 2)
        y_dig = y+((side_len_contour-side_len) // 2)
        img[x_dig:x_dig+side_len, y_dig:y_dig+side_len] = -1  # dig
        img = img.astype(np.int8)
        
        # 2 non-dumpable squares
        x = np.random.randint(0, x_dim - 3 - 1, ())
        y = np.random.randint(0, y_dim - side_len_contour - 1, ())
        dmp[x:x+3, y:y+3] = 0
        x = np.random.randint(0, x_dim - 3 - 1, ())
        y = np.random.randint(0, y_dim - 3 - 1, ())
        dmp[x:x+3, y:y+3] = 0
        dmp = np.where(
            img == 1,
            1,
            dmp
        )
        dmp = dmp.astype(np.bool_)

        # 2 occupation squares
        x = np.random.randint(0, x_dim - 3 - 1, ())
        y = np.random.randint(0, y_dim - 3 - 1, ())
        occ[x:x+3, y:y+3] = 1
        x = np.random.randint(0, x_dim - 3 - 1, ())
        y = np.random.randint(0, y_dim - 3 - 1, ())
        occ[x:x+3, y:y+3] = 1
        occ = np.where(
            (img == 1) | (img == -1),
            0,
            occ,
        )
        occ = occ.astype(np.bool_)

        save_folder_images = Path(save_folder) / "images"
        save_folder_occupancy = Path(save_folder) / "occupancy"
        save_folder_dumpability = Path(save_folder) / "dumpability"
        save_folder_images.mkdir(parents=True, exist_ok=True)
        save_folder_occupancy.mkdir(parents=True, exist_ok=True)
        save_folder_dumpability.mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(save_folder_images, "img_" + str(i)), img)
        np.save(os.path.join(save_folder_occupancy, "img_" + str(i)), occ)
        np.save(os.path.join(save_folder_dumpability, "img_" + str(i)), dmp)
        print(f"Generated squares {i}")


if __name__ == "__main__":
    n_imgs = 100
    x_dim = 60
    y_dim = 60
    side_len = 5
    package_dir = os.path.dirname(os.path.abspath(__file__))
    save_folder = package_dir + f'/../data/openstreet/train/squares_final_{side_len}'
    generate_squares(n_imgs, x_dim, y_dim, side_len, save_folder=save_folder)
