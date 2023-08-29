import numpy as np
import cv2
import os
import json
from pathlib import Path
from digbench.utils import color_dict
from digbench.utils import _get_img_mask
from scipy.signal import convolve2d
import math

def distance_point_to_line(x, y, A, B, C):
    numerator = abs(A*x + B*y + C)
    denominator = math.sqrt(A**2 + B**2)
    distance = numerator / denominator
    return distance


def generate_trenches(level, n_imgs, img_edge_min, img_edge_max, sizes_small, sizes_long, n_edges, min_trench_area_ratio, resolution, option, save_folder=None):
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

        lines_abc = []
        lines_pts = []
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

                # Compute axes
                y_coord = (2 * y + size_y - 1) / 2
                axis_pt1 = (y_coord, x)
                axis_pt2 = (y_coord, x + size_x - 1)
            else:
                size_x = size_small
                size_y = size_long

                # Compute axes
                x_coord = (2 * x + size_x - 1) / 2
                axis_pt1 = (y, x_coord)
                axis_pt2 = (y + size_y - 1, x_coord)
            prev_horizontal = not prev_horizontal
            lines_pts.append([axis_pt1, axis_pt2])
            
            size_x = min(size_x, w-x)
            size_y = min(size_y, h-y)

            if edge_i == 0:
                 # save centroid x, y
                 x_centroid = x + (size_x // 2)
                 y_centroid = y + (size_y // 2)
            
            img[x:x+size_x, y:y+size_y] = np.array(color_dict["digging"])
            
            A = axis_pt2[1] - axis_pt1[1]
            B = axis_pt1[0] - axis_pt2[0]
            C = axis_pt2[0] * axis_pt1[1] - axis_pt1[0] * axis_pt2[1]
            lines_abc.append(
                {
                    "A": float(A),
                    "B": float(B),
                    "C": float(C),
                }
            )

            # if edge_i == n_edges - 1:
            #     agent_x = np.random.randint(0, img.shape[0])
            #     agent_y = np.random.randint(0, img.shape[1])
            #     agent_pos = np.stack((agent_x, agent_y))
            #     print(f"{agent_pos=}")

            #     canvas = img.astype(np.uint8)
            #     cv2.circle(canvas, agent_pos, radius=2, color=(0,0,0), thickness=3)
            #     distances = []
            #     for _, (pts, abc) in enumerate(zip(lines_pts, lines_abc)):
            #         cv2.line(canvas, np.array(pts[0]).astype(np.int32), np.array(pts[1]).astype(np.int32), (0, 0, 0), thickness=2)

            #         distance = distance_point_to_line(agent_pos[0], agent_pos[1], abc["A"], abc["B"], abc["C"])
            #         distances.append(distance)
            #     min_distance = min(distances)
            #     print(f"{min_distance=}")

                # import matplotlib.pyplot as plt
                # plt.imshow(canvas, interpolation="none")
                # plt.show()

            mask = np.zeros_like(img[..., 0], dtype=np.bool_)
            mask[x:x+size_x, y:y+size_y] = np.ones((size_x, size_y), dtype=np.bool_)

        if _get_img_mask(img, color_dict["digging"]).sum() / (w*h) < min_trench_area_ratio:
            print("skipping...")
            continue
        
        # Get dumping constraints
        if level == "medium":
            side_constraints_medium = [
                (np.arange(w) < x_centroid)[:, None].repeat(h, 1),
                (np.arange(w) > x_centroid)[:, None].repeat(h, 1),
                (np.arange(h) < y_centroid)[:, None].repeat(w, 1).T,
                (np.arange(h) > y_centroid)[:, None].repeat(w, 1).T,
            ]
            medium_constraint = side_constraints_medium[i % 4]
            medium_constraint = medium_constraint.astype(np.uint8)
        elif level == "hard":
            side_constraints_hard = [
                (np.arange(w) < x_centroid)[:, None].repeat(h, 1) | (np.arange(h) < y_centroid)[:, None].repeat(w, 1).T,
                (np.arange(w) < x_centroid)[:, None].repeat(h, 1) | (np.arange(h) > y_centroid)[:, None].repeat(w, 1).T,
                (np.arange(w) > x_centroid)[:, None].repeat(h, 1) | (np.arange(h) < y_centroid)[:, None].repeat(w, 1).T,
                (np.arange(w) > x_centroid)[:, None].repeat(h, 1) | (np.arange(h) > y_centroid)[:, None].repeat(w, 1).T,
            ]
            hard_constraint = side_constraints_hard[i % 4]
            hard_constraint = hard_constraint.astype(np.uint8)
        
        
        # Set resolution
        img = cv2.resize(img, (int(img.shape[0] // resolution), int(img.shape[1] // resolution)), interpolation=cv2.INTER_NEAREST)
        if level == "medium":
            medium_constraint = cv2.resize(medium_constraint, (int(medium_constraint.shape[0] // resolution), int(medium_constraint.shape[1] // resolution)), interpolation=cv2.INTER_NEAREST)
        elif level == "hard":
            hard_constraint = cv2.resize(hard_constraint, (int(hard_constraint.shape[0] // resolution), int(hard_constraint.shape[1] // resolution)), interpolation=cv2.INTER_NEAREST)

        img = img.astype(np.uint8)
        i += 1

        # Set contour based on level
        img_black = np.where(
            _get_img_mask(img[..., None].repeat(3, -1), color_dict["neutral"]),
            0,
            img
        )
        kernel_dim = int(min(img_black.shape[:2]) * 0.25)
        kernel = np.ones((kernel_dim, kernel_dim))
        expanded_img = convolve2d(img_black[..., 0], kernel, mode="same")
        contoured_img = np.where(
            (expanded_img > 0) & (img_black[..., 0] == 0),
            1,
            img_black[..., 0]
        )
        tmp = _get_img_mask(contoured_img[..., None].repeat(3, -1), [1, 1, 1])[..., None] * color_dict["dumping"]
        contoured_img = np.where(
            _get_img_mask(contoured_img[..., None].repeat(3, -1), [1, 1, 1])[..., None].repeat(3, -1),
            tmp,
            contoured_img[..., None].repeat(3, -1)
        ).astype(np.uint8)
        contoured_img = np.where(
            contoured_img == 0,
            color_dict["neutral"],
            contoured_img
        ).astype(np.uint8)

        w1, h1, _ = contoured_img.shape
        if level == "easy":
            img = contoured_img
        elif level == "medium":
            img = np.where(
                (_get_img_mask(contoured_img, color_dict["dumping"]) * medium_constraint)[..., None].repeat(3, -1),
                np.array(color_dict["neutral"])[None, None].repeat(w1, 0).repeat(h1, 1),
                contoured_img,
            ).astype(np.uint8)
        elif level == "hard":
            img = np.where(
                (_get_img_mask(contoured_img, color_dict["dumping"]) * hard_constraint)[..., None].repeat(3, -1),
                np.array(color_dict["neutral"])[None, None].repeat(w1, 0).repeat(h1, 1),
                contoured_img,
            ).astype(np.uint8)
        
        if option == 1:
            cv2.imshow("img", img)
            cv2.waitKey(0)
        elif option == 2:
            metadata = {
                "real_dimensions": {"width": float(h), "height": float(w)},
                "axes_ABC": lines_abc
            }

            save_folder_images = Path(save_folder) / "images"
            save_folder_metadata = Path(save_folder) / "metadata"
            save_folder_occupancy = Path(save_folder) / "occupancy"
            save_folder_images.mkdir(parents=True, exist_ok=True)
            save_folder_metadata.mkdir(parents=True, exist_ok=True)
            save_folder_occupancy.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(os.path.join(save_folder_images, "trench_" + str(i) + ".png"), img)
            cv2.imwrite(os.path.join(save_folder_occupancy, "trench_" + str(i) + ".png"), np.ones((img.shape[0], img.shape[1])) * 255)
            with open(os.path.join(save_folder_metadata, "trench_" + str(i) + '.json'), 'w') as outfile:
                        json.dump(metadata, outfile)  # flipped convention
        else:
            raise ValueError(f"Option {option} not supported.")

if __name__ == "__main__":
    n_imgs = 100
    img_edge_min, img_edge_max = 300, 600
    sizes_small = (15, 50)
    sizes_long = (100, 200)
    n_edges = (3, 3)
    min_trench_area_ratio = 0.02
    package_dir = os.path.dirname(os.path.abspath(__file__))
    save_folder = package_dir + '/../data/openstreet/benchmark_' + str(img_edge_min) + '_' + str(img_edge_max)
    generate_trenches("easy", n_imgs, img_edge_min, img_edge_max, sizes_small, sizes_long, n_edges, min_trench_area_ratio, resolution=0.1, option=1, save_folder=save_folder)
