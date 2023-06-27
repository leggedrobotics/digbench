"""
This file contains post processing scripts to generate alternative forms of the standard dataset,
useful if the standard dataset format does not suit a given policy input.

Available transformations:
- to Terra input format (downscaled bitmap)
"""
import os
import cv2
import skimage
import numpy as np
from pathlib import Path
from tqdm import tqdm

def _convert_img_to_terra(img, dig_value, terrain_value, dump_value=None):
    """
    Converts an image from [0, 255] convention
    to [-1, 0, 1] convention.
    """
    img = img.astype(np.int16)
    img = np.where(img == dig_value, -1, img)
    if dump_value is not None:
        img = np.where(img == dump_value, 1, img)
    img = np.where(img == terrain_value, 0, img)
    img = np.where((img != -1) & (img != 1), 0, img)
    return img.astype(np.int8)

def generate_crops_terra(dataset_folder, wm, hm):
    img_folder = Path(dataset_folder) / "crops" / "images"
    destination_folder = Path(dataset_folder) / "terra" / "crops" / f"{wm}x{hm}" /"images"
    destination_folder.mkdir(parents=True, exist_ok=True)
    # metadata_folder = Path(dataset_folder) / "metadata"
    for filename in tqdm(os.listdir(img_folder), desc="crops"):
        file_path = img_folder / filename
        img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)


        # Downsample to (wm, hm)
        img_down = skimage.measure.block_reduce(
                img, (img.shape[0] // wm, img.shape[1] // hm), np.min, cval=255
            )
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        

        # Convert to Terra bitmap convention
        img_terra = _convert_img_to_terra(img_down, dig_value=0, terrain_value=255)
        np.save(destination_folder / filename, img_terra)

def generate_crops_inverted_terra(dataset_folder, wm, hm):
    img_folder = Path(dataset_folder) / "crops_inverted" / "images"
    destination_folder = Path(dataset_folder) / "terra" / "crops_inverted" / f"{wm}x{hm}" /"images"
    destination_folder.mkdir(parents=True, exist_ok=True)
    # metadata_folder = Path(dataset_folder) / "metadata"
    for filename in tqdm(os.listdir(img_folder), desc="crops_inverted"):
        file_path = img_folder / filename
        img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)

        # Downsample to (wm, hm)
        img_down = skimage.measure.block_reduce(
                img, (img.shape[0] // wm, img.shape[1] // hm), np.max, cval=220
            )

        # print(img.shape)
        # print(img[:10, :10])
        # print(img_down.shape)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        # cv2.imshow("img", img_down)
        # cv2.waitKey(0)

        # Convert to Terra bitmap convention
        img_terra = _convert_img_to_terra(img_down, dig_value=220, terrain_value=255)
        np.save(destination_folder / filename, img_terra)

def generate_foundations_terra(dataset_folder, wm, hm):
    img_folder = Path(dataset_folder) / "foundations" / "images"
    destination_folder = Path(dataset_folder) / "terra" / "foundations" / f"{wm}x{hm}" /"images"
    destination_folder.mkdir(parents=True, exist_ok=True)
    # metadata_folder = Path(dataset_folder) / "metadata"
    for filename in tqdm(os.listdir(img_folder), desc="foundations"):
        file_path = img_folder / filename
        img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)


        # Downsample to (wm, hm)
        img_down = skimage.measure.block_reduce(
                img, (img.shape[0] // wm, img.shape[1] // hm), np.min, cval=255
            )
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        # cv2.imshow("img", img_down)
        # cv2.waitKey(0)

        # Convert to Terra bitmap convention
        img_terra = _convert_img_to_terra(img_down, dig_value=0, terrain_value=255)
        np.save(destination_folder / filename, img_terra)

def generate_foundations_inverted_terra(dataset_folder, wm, hm):
    img_folder = Path(dataset_folder) / "foundations_inverted" / "images"
    destination_folder = Path(dataset_folder) / "terra" / "foundations_inverted" / f"{wm}x{hm}" /"images"
    destination_folder.mkdir(parents=True, exist_ok=True)
    # metadata_folder = Path(dataset_folder) / "metadata"
    for filename in tqdm(os.listdir(img_folder), desc="foundations_inverted"):
        file_path = img_folder / filename
        img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)

        # Downsample to (wm, hm)
        img_down = skimage.measure.block_reduce(
                img, (img.shape[0] // wm, img.shape[1] // hm), np.max, cval=220
            )

        # print(img.shape)
        # print(img[:10, :10])
        # print(img_down.shape)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        # cv2.imshow("img", img_down)
        # cv2.waitKey(0)

        # Convert to Terra bitmap convention
        img_terra = _convert_img_to_terra(img_down, dig_value=220, terrain_value=255)
        np.save(destination_folder / filename, img_terra)

def generate_foundations_traversable_terra(dataset_folder, wm, hm):
    print("NOT IMPLEMENTED, SKIPPING")


def generate_dataset_terra_format(dataset_folder, wm, hm, div):
    generate_crops_terra(dataset_folder, wm, hm)
    generate_crops_inverted_terra(dataset_folder, wm, hm)
    generate_foundations_terra(dataset_folder, wm, hm)
    generate_foundations_inverted_terra(dataset_folder, wm, hm)
    generate_foundations_traversable_terra(dataset_folder, wm, hm)  # TODO

    # TODO add others

if __name__ == "__main__":
    # Discretization parameters
    wm, hm = 20, 20  # meters
    div = 10

    package_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_folder = package_dir + '/../data/openstreet/benchmark'
    generate_dataset_terra_format(dataset_folder, wm, hm, div)
