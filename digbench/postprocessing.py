import os
import cv2
import skimage
import json
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


def _convert_all_imgs_to_terra(img_folder, metadata_folder, destination_folder):
    try:
        for i, filename in enumerate(tqdm(os.listdir(img_folder))):
            file_path = img_folder / filename
            img = cv2.imread(str(file_path))
            with open(str(metadata_folder) + f"/{filename.split('.png')[0]}.json") as json_file:
                metadata = json.load(json_file)
            real_w = int(metadata["real_dimensions"]["width"])
            real_h = int(metadata["real_dimensions"]["height"])
            img_downsampled = skimage.measure.block_reduce(
                img, (img.shape[0] // real_w, img.shape[1] // real_h, 1), np.max, cval=255
            )
            img_terra = _convert_img_to_terra(img_downsampled)
            np.save(destination_folder / f"img_{i}", img_terra)
    except Exception as e:
        print(e)
        print("skipping...\n")
        return

def generate_crops_terra(dataset_folder):
    print("Converting crops...")
    for level in ["easy", "medium", "hard"]:
        print(f"    {level}...")
        img_folder = Path(dataset_folder) / "crops" / level / "images"
        metadata_folder = Path(dataset_folder) / "crops" / level / "metadata"
        destination_folder = Path(dataset_folder) / "terra" / "crops" / level / "images"
        destination_folder.mkdir(parents=True, exist_ok=True)
        _convert_all_imgs_to_terra(img_folder, metadata_folder, destination_folder)
    

def generate_foundations_terra(dataset_folder):
    print("Converting foundations...")
    for level in ["easy", "medium", "hard"]:
        print(f"    {level}...")
        img_folder = Path(dataset_folder) / "foundations" / level / "images"
        metadata_folder = Path(dataset_folder) / "foundations" / level / "metadata"
        destination_folder = Path(dataset_folder) / "terra" / "foundations" / level / "images"
        destination_folder.mkdir(parents=True, exist_ok=True)
        _convert_all_imgs_to_terra(img_folder, metadata_folder, destination_folder)


def generate_crops_exterior_terra(dataset_folder):
    print("Converting crops exterior...")
    for level in ["easy", "medium", "hard"]:
        print(f"    {level}...")
        img_folder = Path(dataset_folder) / "crops_exterior" / level / "images"  # TODO is folder name correct?
        metadata_folder = Path(dataset_folder) / "crops_exterior" / level / "metadata"
        destination_folder = Path(dataset_folder) / "terra" / "crops_exterior" / level / "images"  # TODO is folder name correct?
        destination_folder.mkdir(parents=True, exist_ok=True)
        _convert_all_imgs_to_terra(img_folder, metadata_folder, destination_folder)
    

def generate_foundations_exterior_terra(dataset_folder):
    print("Converting foundations exterior...")
    for level in ["easy", "medium", "hard"]:
        print(f"    {level}...")
        img_folder = Path(dataset_folder) / "foundations_exterior" / level / "images"  # TODO is folder name correct?
        metadata_folder = Path(dataset_folder) / "foundations_exterior" / level / "metadata"
        destination_folder = Path(dataset_folder) / "terra" / "foundations_exterior" / level / "images"  # TODO is folder name correct?
        destination_folder.mkdir(parents=True, exist_ok=True)
        _convert_all_imgs_to_terra(img_folder, metadata_folder, destination_folder)


def generate_trenches_terra(dataset_folder):
    print("Converting trenches...")
    for level in ["easy", "medium", "hard"]:
        print(f"    {level}...")
        img_folder = Path(dataset_folder) / "trenches" / level / "images"
        metadata_folder = Path(dataset_folder) / "trenches" / level / "metadata"
        destination_folder = Path(dataset_folder) / "terra" / "trenches" / level / "images"
        destination_folder.mkdir(parents=True, exist_ok=True)
        _convert_all_imgs_to_terra(img_folder, metadata_folder, destination_folder)


def generate_dataset_terra_format(dataset_folder):
    generate_foundations_terra(dataset_folder)
    generate_trenches_terra(dataset_folder)
    generate_crops_terra(dataset_folder)
    generate_crops_exterior_terra(dataset_folder)
    generate_foundations_exterior_terra(dataset_folder)


if __name__ == "__main__":
    sizes = [(20, 40)]#, (40, 80), (80, 160), (160, 320), (320, 640)]
    package_dir = os.path.dirname(os.path.abspath(__file__))
    for size in sizes:
        dataset_folder = package_dir + '/../data/openstreet/train/benchmark_' + str(size[0]) + '_' + str(size[1])
        generate_dataset_terra_format(dataset_folder)
