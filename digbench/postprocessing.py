import os
import cv2
import skimage
import json
import math
import numpy as np
from pathlib import Path
from tqdm import tqdm
from digbench.utils import color_dict
import digbench.utils as utils
from scipy.signal import convolve2d
from digbench.utils import _get_img_mask



def _convert_img_to_terra(img, all_dumpable=False):
    """
    Converts an image from color_dict RGB convention
    to [-1, 0, 1] Terra convention.
    """
    img = img.astype(np.int16)
    img = np.where(img == np.array(color_dict["digging"]), -1, img)
    img = np.where(img == np.array(color_dict["dumping"]), 1, img)
    if all_dumpable:
        img = np.where(img == np.array(color_dict["neutral"]), 1, img)
    else:
        img = np.where(img == np.array(color_dict["neutral"]), 0, img)
    img = np.where((img != -1) & (img != 1), 0, img)
    img = img[..., 0]  # take only 1 channel
    return img.astype(np.int8)

def _convert_occupancy_to_terra(img):
    img = img.astype(np.int16)
    mask = _get_img_mask(img, np.array(color_dict["obstacle"]))
    img = np.where(mask, 1, 0)
    return img.astype(np.bool_)

def _convert_dumpability_to_terra(img):
    img = img.astype(np.int16)
    mask = _get_img_mask(img, np.array(color_dict["nondumpable"]))
    img = np.where(mask, 0, 1)
    return img.astype(np.bool_)

def _generate_border(img):
    """
    img needs to be in Terra conventions and all dumpable
    """
    kernel_dim = int(min(img.shape[:2]) * 0.2)
    kernel = np.ones((kernel_dim, kernel_dim))
    img = img.astype(np.int16)
    img1 = img * 255
    img1 = np.where(
        img1 == 255,
        0,
        img1,
    )
    expanded_img = convolve2d(img1, kernel, mode="same")
    expanded_img = np.where(
        expanded_img == 0,
        1,
        expanded_img,
    )
    expanded_img = np.where(
        (expanded_img < 255) & (expanded_img != 1),
        0,
        expanded_img,
    )
    img = np.where(
        img == 1,
        expanded_img,
        img,
    )
    return img.astype(np.int8)

def _generate_dumping_constraints(img):
    """
    img needs to be in Terra conventions and all dumpable
    """
    kernel_dim = int(min(img.shape[:2]) * 0.4)
    kernel = np.ones((kernel_dim, kernel_dim))
    img = img.astype(np.int16)
    img1 = img * 255
    img1 = np.where(
        img1 == 255,
        0,
        img1,
    )
    expanded_img = convolve2d(img1, kernel, mode="same")
    # expanded_img = np.where(
    #     expanded_img == 0,
    #     1,
    #     expanded_img,
    # )
    expanded_img = np.where(
        expanded_img != 1,
        1,
        expanded_img,
    )
    img = np.where(
        img == -1,
        img,
        expanded_img
    )
    return img.astype(np.int8)


def _convert_all_imgs_to_terra(img_folder, metadata_folder, occupancy_folder, dumpability_folder, destination_folder, size, expansion_factor=1, all_dumpable=False, copy_metadata=True,
                               downsample=False, has_dumpability=True, center_padding=False):
    max_size = size[1]
    # try:
    filename_start = sorted(os.listdir(img_folder))[0].split("_")[0]
    # for i in tqdm(range(len(os.listdir(img_folder)))):
    for i, fn in tqdm(enumerate(os.listdir(img_folder))):

        if i >= 1000:
            break

        # NOTE for the future: os.listdir does not load files in alphabetical order!
        n = int(fn.split(".png")[0].split("_")[1])
        filename = filename_start + f"_{n}.png"
        file_path = img_folder / filename

        occupancy_path = occupancy_folder / filename
        img = cv2.imread(str(file_path))
        occupancy = cv2.imread(str(occupancy_path))

        if has_dumpability:
            dumpability_path = dumpability_folder / filename
            dumpability = cv2.imread(str(dumpability_path))
        # cv2.imshow("dumpability", dumpability)
        # cv2.waitKey(0)

        if downsample:
            with open(str(metadata_folder) + f"/{filename.split('.png')[0]}.json") as json_file:
                metadata = json.load(json_file)
            real_h = int(metadata["real_dimensions"]["width"])  # flipped conventions
            real_w = int(metadata["real_dimensions"]["height"])  # flipped conventions

            img_downsampled = skimage.measure.block_reduce(
                img, (math.ceil(img.shape[0] / real_w), math.ceil(img.shape[1] / real_h), 1), np.max, cval=color_dict["dumping"][0]
            )
            img = img_downsampled
            occupancy_downsampled = skimage.measure.block_reduce(
                occupancy, (math.ceil(occupancy.shape[0] / real_w), math.ceil(occupancy.shape[1] / real_h), 1), np.min, cval=0
            )
            occupancy = occupancy_downsampled
            if has_dumpability:
                dumpability_downsampled = skimage.measure.block_reduce(
                    dumpability, (math.ceil(dumpability.shape[0] / real_w), math.ceil(dumpability.shape[1] / real_h)), np.min, cval=0
                )
                dumpability = dumpability_downsampled



        # assert img_downsampled.shape[:-1] == occupancy_downsampled.shape
        img_terra = _convert_img_to_terra(img, all_dumpable)
        
        # Pad to max size
        if center_padding:
            xdim = max_size-img_terra.shape[0]
            ydim = max_size-img_terra.shape[1]
            # Note: applying full dumping tiles for the centered version
            img_terra_pad = np.ones((max_size, max_size), dtype=img_terra.dtype)
            img_terra_pad[xdim//2:max_size-(xdim-xdim//2), ydim//2:max_size-(ydim-ydim//2)] = img_terra
            # Note: applying no occupancy for the centered version (mismatch with Terra env)
            img_terra_occupancy = np.zeros((max_size, max_size), dtype=np.bool_)
            img_terra_occupancy[xdim//2:max_size-(xdim-xdim//2), ydim//2:max_size-(ydim-ydim//2)] = _convert_occupancy_to_terra(occupancy)
            if has_dumpability:
                img_terra_dumpability = np.zeros((max_size, max_size), dtype=np.bool_)
                img_terra_dumpability[xdim//2:max_size-(xdim-xdim//2), ydim//2:max_size-(ydim-ydim//2)] = _convert_dumpability_to_terra(dumpability)
        else:
            img_terra_pad = np.zeros((max_size, max_size), dtype=img_terra.dtype)
            img_terra_pad[:img_terra.shape[0], :img_terra.shape[1]] = img_terra
            img_terra_occupancy = np.ones((max_size, max_size), dtype=np.bool_)
            img_terra_occupancy[:occupancy.shape[0], :occupancy.shape[1]] = _convert_occupancy_to_terra(occupancy)
            if has_dumpability:
                img_terra_dumpability = np.zeros((max_size, max_size), dtype=np.bool_)
                img_terra_dumpability[:dumpability.shape[0], :dumpability.shape[1]] = _convert_dumpability_to_terra(dumpability)

        
        destination_folder_images = destination_folder / "images"
        destination_folder_occupancy = destination_folder / "occupancy"
        destination_folder_images.mkdir(parents=True, exist_ok=True)
        destination_folder_occupancy.mkdir(parents=True, exist_ok=True)
        destination_folder_dumpability = destination_folder / "dumpability"
        destination_folder_dumpability.mkdir(parents=True, exist_ok=True)
        if copy_metadata:
            destination_folder_metadata = destination_folder / "metadata"
            destination_folder_metadata.mkdir(parents=True, exist_ok=True)

        img_terra_pad = img_terra_pad.repeat(expansion_factor, 0).repeat(expansion_factor, 1)
        img_terra_occupancy = img_terra_occupancy.repeat(expansion_factor, 0).repeat(expansion_factor, 1)
        if has_dumpability:
            img_terra_dumpability = img_terra_dumpability.repeat(expansion_factor, 0).repeat(expansion_factor, 1)
            
        np.save(destination_folder_images / f"img_{i+1}", img_terra_pad)
        np.save(destination_folder_occupancy / f"img_{i+1}", img_terra_occupancy)
        if has_dumpability:
            np.save(destination_folder_dumpability / f"img_{i+1}", img_terra_dumpability)
        else:
            np.save(destination_folder_dumpability / f"img_{i+1}", np.ones_like(img_terra_pad))


    if copy_metadata:
        utils.copy_metadata(str(metadata_folder), str(destination_folder_metadata))
    # except Exception as e:
    #     print(e)
    #     print("skipping...\n")
    #     return

def generate_crops_terra(dataset_folder, size):
    print("Converting crops...")
    for level in ["easy", "medium", "hard"]:
        print(f"    {level}...")
        img_folder = Path(dataset_folder) / "crops" / level / "images"
        metadata_folder = Path(dataset_folder) / "crops" / level / "metadata"
        occupancy_folder = Path(dataset_folder) / "crops" / level / "occupancy"
        dumpability_folder = Path(dataset_folder) / "crops" / level / "dumpability"
        destination_folder = Path(dataset_folder) / "terra" / "crops" / level
        destination_folder.mkdir(parents=True, exist_ok=True)
        _convert_all_imgs_to_terra(img_folder, metadata_folder, occupancy_folder, dumpability_folder, destination_folder, size)
    

def generate_foundations_terra(dataset_folder, size, all_dumpable, copy_metadata):
    print("Converting foundations...")
    img_folder = Path(dataset_folder) / "foundations"  / "images"
    metadata_folder = Path(dataset_folder) / "foundations"  / "metadata"
    occupancy_folder = Path(dataset_folder) / "foundations"  / "occupancy"
    dumpability_folder = Path(dataset_folder) / "foundations"  / "dumpability"
    destination_folder = Path(dataset_folder) / "terra" / "foundations" 
    destination_folder.mkdir(parents=True, exist_ok=True)
    _convert_all_imgs_to_terra(img_folder, metadata_folder, occupancy_folder, dumpability_folder, destination_folder, size, all_dumpable=all_dumpable, copy_metadata=copy_metadata,
                               downsample=True, has_dumpability=False, center_padding=True)
    for level in ["easy", "medium", "hard"]:
        print(f"    {level}...")
        img_folder = Path(dataset_folder) / "foundations" / level / "images"
        metadata_folder = Path(dataset_folder) / "foundations" / level / "metadata"
        occupancy_folder = Path(dataset_folder) / "foundations" / level / "occupancy"
        dumpability_folder = Path(dataset_folder) / "foundations" / level / "dumpability"
        destination_folder = Path(dataset_folder) / "terra" / "foundations" / level
        destination_folder.mkdir(parents=True, exist_ok=True)
        _convert_all_imgs_to_terra(img_folder, metadata_folder, occupancy_folder, dumpability_folder, destination_folder, size, all_dumpable=all_dumpable, copy_metadata=copy_metadata,
                                   )


def generate_crops_exterior_terra(dataset_folder, size):
    print("Converting crops exterior...")
    for level in ["easy", "medium", "hard"]:
        print(f"    {level}...")
        img_folder = Path(dataset_folder) / "crops_exterior" / level / "images"  # TODO is folder name correct?
        metadata_folder = Path(dataset_folder) / "crops_exterior" / level / "metadata"
        occupancy_folder = Path(dataset_folder) / "crops_exterior" / level / "occupancy"
        dumpability_folder = Path(dataset_folder) / "crops_exterior" / level / "dumpability"
        destination_folder = Path(dataset_folder) / "terra" / "crops_exterior" / level  # TODO is folder name correct?
        destination_folder.mkdir(parents=True, exist_ok=True)
        _convert_all_imgs_to_terra(img_folder, metadata_folder, occupancy_folder, dumpability_folder, destination_folder, size)
    

def generate_foundations_exterior_terra(dataset_folder, size):
    print("Converting foundations exterior...")
    for level in ["easy", "medium", "hard"]:
        print(f"    {level}...")
        img_folder = Path(dataset_folder) / "foundations_exterior" / level / "images"  # TODO is folder name correct?
        metadata_folder = Path(dataset_folder) / "foundations_exterior" / level / "metadata"
        occupancy_folder = Path(dataset_folder) / "foundations_exterior" / level / "occupancy"
        dumpability_folder = Path(dataset_folder) / "foundations_exterior" / level / "dumpability"
        destination_folder = Path(dataset_folder) / "terra" / "foundations_exterior" / level  # TODO is folder name correct?
        destination_folder.mkdir(parents=True, exist_ok=True)
        _convert_all_imgs_to_terra(img_folder, metadata_folder, occupancy_folder, dumpability_folder, destination_folder, size)


def generate_trenches_terra(dataset_folder, size, expansion_factor, all_dumpable):
    print("Converting trenches...")
    # for level in ["lev1-T-trenches-contour", "lev2-T-trenches-contour", "lev3-T-trenches-contour", "lev4-T-trenches-contour"]:
    for level in ["easy", "medium", "hard"]:
        print(f"    {level}...")
        img_folder = Path(dataset_folder) / "trenches" / level / "images"
        metadata_folder = Path(dataset_folder) / "trenches" / level / "metadata"
        occupancy_folder = Path(dataset_folder) / "trenches" / level / "occupancy"
        dumpability_folder = Path(dataset_folder) / "trenches" / level / "dumpability"
        destination_folder = Path(dataset_folder) / "terra" / "trenches" / level
        destination_folder.mkdir(parents=True, exist_ok=True)
        _convert_all_imgs_to_terra(img_folder, metadata_folder, occupancy_folder, dumpability_folder, destination_folder, size,expansion_factor=expansion_factor, all_dumpable=all_dumpable)


def generate_dataset_terra_format(dataset_folder, size):
    generate_foundations_terra(dataset_folder, size, all_dumpable=False, copy_metadata=False)
    generate_trenches_terra(dataset_folder, size, expansion_factor=1, all_dumpable=False)
    generate_crops_terra(dataset_folder, size)
    generate_crops_exterior_terra(dataset_folder, size)
    generate_foundations_exterior_terra(dataset_folder, size)


if __name__ == "__main__":
    sizes = [(60, 60)]#, (40, 80), (80, 160), (160, 320), (320, 640)]
    package_dir = os.path.dirname(os.path.abspath(__file__))
    for size in sizes:
        # dataset_folder = package_dir + '/../data/openstreet/train/benchmark_60_61'
        dataset_folder = package_dir + '/../data/openstreet/train/benchmark_20_50'
        generate_dataset_terra_format(dataset_folder, size)
