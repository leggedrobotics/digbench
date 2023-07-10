import os
import cv2
import skimage
import json
import math
import numpy as np
from pathlib import Path
from tqdm import tqdm
from digbench.utils import color_dict


def _convert_img_to_terra(img):
    """
    Converts an image from color_dict RGB convention
    to [-1, 0, 1] Terra convention.
    """
    img = img.astype(np.int16)
    img = np.where(img == np.array(color_dict["digging"]), -1, img)
    img = np.where(img == np.array(color_dict["dumping"]), 1, img)
    img = np.where(img == np.array(color_dict["neutral"]), 0, img)
    img = np.where((img != -1) & (img != 1), 0, img)
    img = img[..., 0]  # take only 1 channel
    return img.astype(np.int8)

def _convert_occupancy_to_terra(img):
    img = img.astype(np.int16)
    img = np.where(img == 0, 1, 0)
    return img.astype(np.int8)

def _convert_all_imgs_to_terra(img_folder, metadata_folder, occupancy_folder, destination_folder, size):
    max_size = size[1]
    try:
        for i, filename in enumerate(tqdm(os.listdir(img_folder))):
            file_path = img_folder / filename
            occupancy_path = occupancy_folder / filename
            img = cv2.imread(str(file_path))
            occupancy = cv2.imread(str(occupancy_path), cv2.IMREAD_GRAYSCALE)
            with open(str(metadata_folder) + f"/{filename.split('.png')[0]}.json") as json_file:
                metadata = json.load(json_file)
            real_h = int(metadata["real_dimensions"]["width"])  # flipped conventions
            real_w = int(metadata["real_dimensions"]["height"])  # flipped conventions
            img_downsampled = skimage.measure.block_reduce(
                img, (math.ceil(img.shape[0] / real_w), math.ceil(img.shape[1] / real_h), 1), np.max, cval=color_dict["neutral"][0]
            )
            occupancy_downsampled = skimage.measure.block_reduce(
                occupancy, (math.ceil(occupancy.shape[0] / real_w), math.ceil(occupancy.shape[1] / real_h)), np.min, cval=0
            )
            img_terra = _convert_img_to_terra(img_downsampled)
            
            # Pad to max size
            img_terra_pad = np.zeros((max_size, max_size), dtype=img_terra.dtype)
            img_terra_pad[:img_terra.shape[0], :img_terra.shape[1]] = img_terra
            img_terra_occupancy = np.ones((max_size, max_size), dtype=img_terra.dtype)
            img_terra_occupancy[:occupancy_downsampled.shape[0], :occupancy_downsampled.shape[1]] = _convert_occupancy_to_terra(occupancy_downsampled)

            destination_folder_images = destination_folder / "images"
            destination_folder_occupancy = destination_folder / "occupancy"
            destination_folder_images.mkdir(parents=True, exist_ok=True)
            destination_folder_occupancy.mkdir(parents=True, exist_ok=True)
            np.save(destination_folder_images / f"img_{i}", img_terra_pad)
            # np.save(destination_folder_images / filename[:-4], img_terra_pad)
            np.save(destination_folder_occupancy / f"img_{i}", img_terra_occupancy)
            # np.save(destination_folder_occupancy / filename[:-4], img_terra_occupancy)
    except Exception as e:
        print(e)
        print("skipping...\n")
        return

def generate_crops_terra(dataset_folder, size):
    print("Converting crops...")
    for level in ["easy", "medium", "hard"]:
        print(f"    {level}...")
        img_folder = Path(dataset_folder) / "crops" / level / "images"
        metadata_folder = Path(dataset_folder) / "crops" / level / "metadata"
        occupancy_folder = Path(dataset_folder) / "crops" / level / "occupancy"
        destination_folder = Path(dataset_folder) / "terra" / "crops" / level
        destination_folder.mkdir(parents=True, exist_ok=True)
        _convert_all_imgs_to_terra(img_folder, metadata_folder, occupancy_folder, destination_folder, size)
    

def generate_foundations_terra(dataset_folder, size):
    print("Converting foundations...")
    for level in ["easy", "medium", "hard"]:
        print(f"    {level}...")
        img_folder = Path(dataset_folder) / "foundations" / level / "images"
        metadata_folder = Path(dataset_folder) / "foundations" / level / "metadata"
        occupancy_folder = Path(dataset_folder) / "foundations" / level / "occupancy"
        destination_folder = Path(dataset_folder) / "terra" / "foundations" / level
        destination_folder.mkdir(parents=True, exist_ok=True)
        _convert_all_imgs_to_terra(img_folder, metadata_folder, occupancy_folder, destination_folder, size)


def generate_crops_exterior_terra(dataset_folder, size):
    print("Converting crops exterior...")
    for level in ["easy", "medium", "hard"]:
        print(f"    {level}...")
        img_folder = Path(dataset_folder) / "crops_exterior" / level / "images"  # TODO is folder name correct?
        metadata_folder = Path(dataset_folder) / "crops_exterior" / level / "metadata"
        occupancy_folder = Path(dataset_folder) / "crops_exterior" / level / "occupancy"
        destination_folder = Path(dataset_folder) / "terra" / "crops_exterior" / level  # TODO is folder name correct?
        destination_folder.mkdir(parents=True, exist_ok=True)
        _convert_all_imgs_to_terra(img_folder, metadata_folder, occupancy_folder, destination_folder, size)
    

def generate_foundations_exterior_terra(dataset_folder, size):
    print("Converting foundations exterior...")
    for level in ["easy", "medium", "hard"]:
        print(f"    {level}...")
        img_folder = Path(dataset_folder) / "foundations_exterior" / level / "images"  # TODO is folder name correct?
        metadata_folder = Path(dataset_folder) / "foundations_exterior" / level / "metadata"
        occupancy_folder = Path(dataset_folder) / "foundations_exterior" / level / "occupancy"
        destination_folder = Path(dataset_folder) / "terra" / "foundations_exterior" / level  # TODO is folder name correct?
        destination_folder.mkdir(parents=True, exist_ok=True)
        _convert_all_imgs_to_terra(img_folder, metadata_folder, occupancy_folder, destination_folder, size)


def generate_trenches_terra(dataset_folder, size):
    print("Converting trenches...")
    for level in ["easy", "medium", "hard"]:
        print(f"    {level}...")
        img_folder = Path(dataset_folder) / "trenches" / level / "images"
        metadata_folder = Path(dataset_folder) / "trenches" / level / "metadata"
        occupancy_folder = Path(dataset_folder) / "trenches" / level / "occupancy"
        destination_folder = Path(dataset_folder) / "terra" / "trenches" / level
        destination_folder.mkdir(parents=True, exist_ok=True)
        _convert_all_imgs_to_terra(img_folder, metadata_folder, occupancy_folder, destination_folder, size)


def generate_dataset_terra_format(dataset_folder, size):
    generate_foundations_terra(dataset_folder, size)
    generate_trenches_terra(dataset_folder, size)
    # generate_crops_terra(dataset_folder, size)
    # generate_crops_exterior_terra(dataset_folder, size)
    # generate_foundations_exterior_terra(dataset_folder, size)


if __name__ == "__main__":
    sizes = [(20, 40)]#, (40, 80), (80, 160), (160, 320), (320, 640)]
    package_dir = os.path.dirname(os.path.abspath(__file__))
    for size in sizes:
        dataset_folder = package_dir + '/../data/openstreet/train/benchmark_' + str(size[0]) + '_' + str(size[1])
        generate_dataset_terra_format(dataset_folder, size)
