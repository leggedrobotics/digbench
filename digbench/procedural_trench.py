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

    n_obs_min, n_obs_max = 1, 3
    size_obstacle_min, size_obstacle_max = 2, 8

    n_nodump_min, n_nodump_max = 1, 3
    size_nodump_min, size_nodump_max = 2, 8
    
    i = 0
    while i < n_imgs:
        
        not_good_trench = True
        while not_good_trench:
            w, h = np.random.randint(img_edge_min, img_edge_max, (2,), dtype=np.int32)
            img = np.ones((w, h, 3)) * np.array(color_dict["neutral"])

            corner_dump = np.random.randint(0, 4, ())
            if corner_dump == 0:
                img[0:int(0.8*w), :h, :] = np.ones((int(w*0.8), h, 3)) * np.array(color_dict["dumping"])
            elif corner_dump == 1:
                img[int(0.2*w):, :h, :] = np.ones((w-int(w*0.2), h, 3)) * np.array(color_dict["dumping"])
            elif corner_dump == 2:
                img[:w, int(0.2*h):, :] = np.ones((w, h-int(0.2*h), 3)) * np.array(color_dict["dumping"])
            elif corner_dump == 3:
                img[:w, :int(0.8*h), :] = np.ones((w, int(0.8*h), 3)) * np.array(color_dict["dumping"])

            n_edges = np.random.randint(min_edges, max_edges + 1)

            prev_horizontal = True if np.random.choice([0,1]).astype(np.bool_) else False

            lines_abc = []
            lines_pts = []
            for edge_i in range(n_edges):
                if edge_i == 0:
                    mask = np.ones_like(img[..., 0], dtype=np.bool_)
                    cumulative_mask = np.zeros_like(img[..., 0], dtype=np.bool_)
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
                    axis_pt1 = (float(y_coord), float(x))
                    axis_pt2 = (float(y_coord), float(x) + size_x - 1)
                else:
                    size_x = size_small
                    size_y = size_long

                    # Compute axes
                    x_coord = (2 * x + size_x - 1) / 2
                    axis_pt1 = (float(y), float(x_coord))
                    axis_pt2 = (float(y) + size_y - 1, float(x_coord))
                prev_horizontal = not prev_horizontal
                lines_pts.append([axis_pt1, axis_pt2])
                # print(f"{axis_pt1=}, {axis_pt2=}")
                
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
                #     cv2.circle(canvas, agent_pos, radius=2, color=(0,0,0), thickness=1)
                #     distances = []
                #     for _, (pts, abc) in enumerate(zip(lines_pts, lines_abc)):
                #         cv2.line(canvas, np.array(pts[0]).astype(np.int32), np.array(pts[1]).astype(np.int32), (0, 0, 0), thickness=1)

                #         distance = distance_point_to_line(agent_pos[0], agent_pos[1], abc["A"], abc["B"], abc["C"])
                #         distances.append(distance)
                #     min_distance = min(distances)
                #     print(f"{min_distance=}")

                #     import matplotlib.pyplot as plt
                #     plt.imshow(canvas, interpolation="none")
                #     plt.show()

                mask = np.zeros_like(img[..., 0], dtype=np.bool_)
                mask[x:x+size_x, y:y+size_y] = np.ones((size_x, size_y), dtype=np.bool_)
                cumulative_mask = cumulative_mask | mask

            ixts = img.shape[0]
            iyts = img.shape[1]
            # Set margin % here
            ixt = int(ixts * 0.15)
            iyt = int(iyts * 0.15)
            img_test = img.copy()
            img_test[ixt:ixts-ixt, iyt:iyts-iyt] = np.array(color_dict["neutral"])
            if np.any(_get_img_mask(img_test, color_dict["digging"])):
                print("Trench on the border, skipping...")
                continue
            else:
                break

        # continue

        if _get_img_mask(img, color_dict["digging"]).sum() / (w*h) < min_trench_area_ratio:
            print("skipping...")
            continue

        # Obstacles 
        n_occ = 0
        occ = np.ones_like(img) * 255
        n_obs_now = np.random.randint(n_obs_min, n_obs_max + 1, ())
        while n_occ < n_obs_now:
            sizeox = np.random.randint(size_obstacle_min, size_obstacle_max + 1, ())
            sizeoy = np.random.randint(size_obstacle_min, size_obstacle_max + 1, ())
            x = np.random.randint(0, w - sizeox - 1, ())
            y = np.random.randint(0, h - sizeoy - 1, ())
            if cumulative_mask[x:x+sizeox, y:y+sizeoy].sum() == 0:
                occ[x:x+sizeox, y:y+sizeoy] = np.ones((3,)) * color_dict["obstacle"]
                n_occ += 1

        # Non-dumpable but traversable         
        n_dmp = 0
        dmp = np.ones_like(img) * 255
        n_obs_now = np.random.randint(n_nodump_min, n_nodump_max + 1, ())
        mask_occ = _get_img_mask(occ, color_dict["obstacle"])
        while n_dmp < n_obs_now:
            dmp_type = np.random.randint(0, 2, ())
            if dmp_type.item() == 0:
                # Squares
                sizeox = np.random.randint(size_nodump_min, size_nodump_max + 1, ())
                sizeoy = np.random.randint(size_nodump_min, size_nodump_max + 1, ())
                x = np.random.randint(0, w - sizeox - 1, ())
                y = np.random.randint(0, h - sizeoy - 1, ())
            elif dmp_type.item() == 1:
                # Roads
                road_direction = np.random.randint(0, 2, ())
                if road_direction.item() == 0:
                    sizeox = w
                    sizeoy = np.random.randint(size_nodump_min, size_nodump_max + 1, ())
                    x = 0
                    y = np.random.randint(0, h - sizeoy - 1, ())
                elif road_direction.item() == 1:
                    sizeox = np.random.randint(size_nodump_min, size_nodump_max + 1, ())
                    sizeoy = h
                    x = np.random.randint(0, w - sizeox - 1, ())
                    y = 0
            else:
                raise(ValueError(f"{dmp_type.item()=}"))
            
            if cumulative_mask[x:x+sizeox, y:y+sizeoy].sum() == 0 and mask_occ[x:x+sizeox, y:y+sizeoy].sum() == 0:
                dmp[x:x+sizeox, y:y+sizeoy] = np.ones((3,)) * color_dict["nondumpable"]
                n_dmp += 1

        # Get dumping constraints
        # if level == "medium":
        #     side_constraints_medium = [
        #         (np.arange(w) < x_centroid)[:, None].repeat(h, 1),
        #         (np.arange(w) > x_centroid)[:, None].repeat(h, 1),
        #         (np.arange(h) < y_centroid)[:, None].repeat(w, 1).T,
        #         (np.arange(h) > y_centroid)[:, None].repeat(w, 1).T,
        #     ]
        #     medium_constraint = side_constraints_medium[i % 4]
        #     medium_constraint = medium_constraint.astype(np.uint8)
        # elif level == "hard":
        #     side_constraints_hard = [
        #         (np.arange(w) < x_centroid)[:, None].repeat(h, 1) | (np.arange(h) < y_centroid)[:, None].repeat(w, 1).T,
        #         (np.arange(w) < x_centroid)[:, None].repeat(h, 1) | (np.arange(h) > y_centroid)[:, None].repeat(w, 1).T,
        #         (np.arange(w) > x_centroid)[:, None].repeat(h, 1) | (np.arange(h) < y_centroid)[:, None].repeat(w, 1).T,
        #         (np.arange(w) > x_centroid)[:, None].repeat(h, 1) | (np.arange(h) > y_centroid)[:, None].repeat(w, 1).T,
        #     ]
        #     hard_constraint = side_constraints_hard[i % 4]
        #     hard_constraint = hard_constraint.astype(np.uint8)
        
        
        # Set resolution
        img = cv2.resize(img, (int(img.shape[0] // resolution), int(img.shape[1] // resolution)), interpolation=cv2.INTER_NEAREST)
        # if level == "medium":
        #     medium_constraint = cv2.resize(medium_constraint, (int(medium_constraint.shape[0] // resolution), int(medium_constraint.shape[1] // resolution)), interpolation=cv2.INTER_NEAREST)
        # elif level == "hard":
        #     hard_constraint = cv2.resize(hard_constraint, (int(hard_constraint.shape[0] // resolution), int(hard_constraint.shape[1] // resolution)), interpolation=cv2.INTER_NEAREST)

        img = img.astype(np.uint8)
        i += 1

        # Set contour based on level
        # img_black = np.where(
        #     _get_img_mask(img[..., None].repeat(3, -1), color_dict["neutral"]),
        #     0,
        #     img
        # )
        # kernel_dim = int(min(img_black.shape[:2]) * 0.4)
        # kernel = np.ones((kernel_dim, kernel_dim))
        # expanded_img = convolve2d(img_black[..., 0], kernel, mode="same")
        # contoured_img = np.where(
        #     (expanded_img > 0) & (img_black[..., 0] == 0),
        #     1,
        #     img_black[..., 0]
        # )
        # tmp = _get_img_mask(contoured_img[..., None].repeat(3, -1), [1, 1, 1])[..., None] * color_dict["dumping"]
        # contoured_img = np.where(
        #     _get_img_mask(contoured_img[..., None].repeat(3, -1), [1, 1, 1])[..., None].repeat(3, -1),
        #     tmp,
        #     contoured_img[..., None].repeat(3, -1)
        # ).astype(np.uint8)
        # contoured_img_dumping = np.where(
        #     contoured_img == 0,
        #     color_dict["neutral"],
        #     contoured_img
        # ).astype(np.uint8)

        # # Set neutral contour
        # img_black = np.where(
        #     _get_img_mask(img[..., None].repeat(3, -1), color_dict["neutral"]),
        #     0,
        #     img
        # )
        # kernel_dim = int(min(img_black.shape[:2]) * 0.13)
        # kernel = np.ones((kernel_dim, kernel_dim))
        # expanded_img = convolve2d(img_black[..., 0], kernel, mode="same")
        # contoured_img = np.where(
        #     (expanded_img > 0) & (img_black[..., 0] == 0),
        #     1,
        #     img_black[..., 0]
        # )
        # tmp = _get_img_mask(contoured_img[..., None].repeat(3, -1), [1, 1, 1])[..., None] * ([el + 1 for el in color_dict["neutral"]])
        # contoured_img = np.where(
        #     _get_img_mask(contoured_img[..., None].repeat(3, -1), [1, 1, 1])[..., None].repeat(3, -1),
        #     tmp,
        #     contoured_img[..., None].repeat(3, -1)
        # ).astype(np.uint8)
        # contoured_img_neutral = np.where(
        #     contoured_img == 0,
        #     color_dict["neutral"],
        #     contoured_img
        # ).astype(np.uint8)
        # w1, h1, _ = contoured_img.shape
        # if level == "easy":
        #     img = np.where(
        #         (_get_img_mask(contoured_img_neutral, [el + 1 for el in color_dict["neutral"]]))[..., None].repeat(3, -1),
        #         contoured_img_neutral,
        #         contoured_img_dumping
        #     )
        # elif level == "medium":
        #     contoured_img_dumping = np.where(
        #         (_get_img_mask(contoured_img_dumping, color_dict["dumping"]) * medium_constraint)[..., None].repeat(3, -1),
        #         np.array(color_dict["neutral"])[None, None].repeat(w1, 0).repeat(h1, 1),
        #         contoured_img_dumping,
        #     ).astype(np.uint8)
        #     img = np.where(
        #         (_get_img_mask(contoured_img_neutral, [el + 1 for el in color_dict["neutral"]]))[..., None].repeat(3, -1),
        #         contoured_img_neutral,
        #         contoured_img_dumping
        #     )
        # elif level == "hard":
        #     contoured_img_dumping = np.where(
        #         (_get_img_mask(contoured_img_dumping, color_dict["dumping"]) * hard_constraint)[..., None].repeat(3, -1),
        #         np.array(color_dict["neutral"])[None, None].repeat(w1, 0).repeat(h1, 1),
        #         contoured_img_dumping,
        #     ).astype(np.uint8)
        #     img = np.where(
        #         (_get_img_mask(contoured_img_neutral, [el + 1 for el in color_dict["neutral"]]))[..., None].repeat(3, -1),
        #         contoured_img_neutral,
        #         contoured_img_dumping
        #     )
        
        if option == 1:
            cv2.imshow("img", img)
            cv2.waitKey(0)
        elif option == 2:
            metadata = {
                "real_dimensions": {"width": float(h), "height": float(w)},
                "axes_ABC": lines_abc,
                "lines_pts": lines_pts,
            }

            save_folder_images = Path(save_folder) / "images"
            save_folder_metadata = Path(save_folder) / "metadata"
            save_folder_occupancy = Path(save_folder) / "occupancy"
            save_folder_dumpability = Path(save_folder) / "dumpability"
            save_folder_images.mkdir(parents=True, exist_ok=True)
            save_folder_metadata.mkdir(parents=True, exist_ok=True)
            save_folder_occupancy.mkdir(parents=True, exist_ok=True)
            save_folder_dumpability.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(os.path.join(save_folder_images, "trench_" + str(i) + ".png"), img)
            cv2.imwrite(os.path.join(save_folder_occupancy, "trench_" + str(i) + ".png"), occ)
            cv2.imwrite(os.path.join(save_folder_dumpability, "trench_" + str(i) + ".png"), dmp)
            with open(os.path.join(save_folder_metadata, "trench_" + str(i) + '.json'), 'w') as outfile:
                json.dump(metadata, outfile)  # flipped convention
            print(f"Generated trench {i}")
        else:
            raise ValueError(f"Option {option} not supported.")

if __name__ == "__main__":
    n_imgs = 100
    img_edge_min, img_edge_max = 300, 600
    sizes_small = (15, 50)
    sizes_long = (100, 200)
    n_edges = (2, 2)
    min_trench_area_ratio = 0.02
    package_dir = os.path.dirname(os.path.abspath(__file__))
    save_folder = package_dir + '/../data/openstreet/benchmark_' + str(img_edge_min) + '_' + str(img_edge_max)
    generate_trenches("easy", n_imgs, img_edge_min, img_edge_max, sizes_small, sizes_long, n_edges, min_trench_area_ratio, resolution=0.1, option=1, save_folder=save_folder)
