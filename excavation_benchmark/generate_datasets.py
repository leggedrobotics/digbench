from coverage_planning_ros import openstreet_plugin
from coverage_planning_ros import helpers
import shutil
import os
# set seed


def download_city_crops(main_folder):
    center_bbox = (47.378177, 47.364622, 8.526535, 8.544894)
    folder_path = main_folder + '/crops_raw'
    # download images if they don't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        num_crop = 100
        current_crops = 0
        while current_crops < num_crop:
            openstreet_plugin.collect_random_crops(center_bbox, 0.02, folder_path)
            # read in the folder how many png images there are
            current_crops = len([name for name in os.listdir(folder_path) if name.endswith(".png")])

    # change resolution
    image_folder = main_folder + '/crops_raw'
    metadata_folder = main_folder + '/crops_raw'
    image_resized_folder = main_folder + '/crops_resized'
    resolution = 1.0
    helpers.preprocess_dataset_fixed_resolution(image_folder, metadata_folder, image_resized_folder, resolution,
                                             flip=False)

    # pad images
    image_folder = main_folder + '/crops_resized'
    save_folder = main_folder + '/crops_padded'
    metadata_folder = main_folder + '/crops_raw'
    padding = 5
    helpers.pad_images_and_update_metadata(image_folder, metadata_folder, padding, (255, 255, 255), save_folder)

    # fill in the holes
    image_folder = main_folder + '/crops_padded'
    dataset_folder = main_folder + '/exterior_crops'
    # make it if it doesn't exist
    os.makedirs(dataset_folder, exist_ok=True)

    save_folder = dataset_folder + "/images"
    helpers.fill_dataset(image_folder, save_folder, copy_metadata=False)
    # copy metadata
    helpers.copy_metadata(main_folder + '/crops_padded', main_folder + '/exterior_crops/metadata')
    # make occupancy, in this case is the same as the images folder
    # copy folder but change name
    shutil.copytree(main_folder + '/exterior_crops/images', main_folder + '/exterior_crops/occupancy',
                    dirs_exist_ok=True)


def create_city_crops_inverted(main_folder):
    # invert the images
    image_folder = main_folder + '/exterior_crops/images'
    dataset_folder = main_folder + '/crops'
    # make it if it doesn't exist
    os.makedirs(dataset_folder, exist_ok=True)
    save_folder = dataset_folder + "/images"
    helpers.invert_dataset(image_folder, save_folder)
    # copy metadata
    helpers.copy_metadata(main_folder + '/exterior_crops/metadata', main_folder + '/crops/metadata')
    # create_emtpy_occupancy
    helpers.generate_empty_occupancy(save_folder, dataset_folder + "/occupancy")


def download_foundations(main_folder):
    center_bbox = (47.378177, 47.364622, 8.526535, 8.544894)
    dataset_folder = main_folder + '/foundations_raw'
    # if it does not exist
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
        openstreet_plugin.get_building_shapes_from_OSM(*center_bbox, option=2,
                                                       save_folder=dataset_folder)
    # filter out small cases
    image_folder = main_folder + '/foundations_raw/images'
    save_folder = main_folder + '/foundations_filtered/images'
    metadata_folder = main_folder + '/foundations_raw/metadata'
    min_size = (20, 20)
    helpers.size_filter(image_folder, save_folder, metadata_folder, min_size)

    # pad the edges
    image_folder = main_folder + '/foundations_filtered/images'
    save_folder = main_folder + '/foundations_filtered_padded'
    metadata_folder = main_folder + '/foundations_raw/metadata'
    padding = 5
    helpers.pad_images_and_update_metadata(image_folder, metadata_folder, padding, (255, 255, 255), save_folder)

    # set resolution
    image_folder = main_folder + '/foundations_filtered_padded'
    metadata_folder = main_folder + '/foundations_filtered_padded'
    image_resized_folder = main_folder + '/foundations_filtered_padded_resized'
    resolution = 0.2
    helpers.preprocess_dataset_fixed_resolution(image_folder, metadata_folder, image_resized_folder, resolution)


def create_foundations(main_folder):
    # fill holes
    image_folder = main_folder + '/foundations_filtered_padded_resized'
    dataset_folder = main_folder + '/exterior_foundations'
    # make it if it doesn't exist
    os.makedirs(dataset_folder, exist_ok=True)
    save_folder = dataset_folder + "/images"
    metadata_folder = main_folder + '/foundations_filtered_padded'
    helpers.fill_dataset(image_folder, save_folder, copy_metadata=False)
    # copy metadata folder to save folder and change its name to metadata
    helpers.copy_metadata(metadata_folder, dataset_folder + '/metadata')
    # make occupancy, in this case is the same as the images folder
    # copy folder but change name
    shutil.copytree(main_folder + '/exterior_foundations/images', main_folder + '/exterior_foundations/occupancy', dirs_exist_ok=True)


def create_foundations_traversable(main_folder):
    dataset_folder = main_folder + '/exterior_foundations'
    save_folder = main_folder + '/exterior_foundations_traversable'
    helpers.make_obstacles_traversable(dataset_folder + '/images', save_folder + '/images')
    # copy metadata folder to save folder and change its name to metadata
    helpers.copy_metadata(dataset_folder + '/metadata', save_folder + '/metadata')
    # generate empty occupancy
    helpers.generate_empty_occupancy(dataset_folder + '/images', save_folder + "/occupancy")


def create_foundations_inverted(main_folder):
    dataset_folder = main_folder + '/foundations_filtered_padded_resized'
    save_folder = main_folder + '/foundations'
    inverted_image_folder = save_folder + '/images'
    helpers.invert_dataset(dataset_folder, inverted_image_folder)
    # copy metadata folder to save folder and change its name to metadata
    helpers.copy_metadata(dataset_folder, save_folder + '/metadata')
    # generate empty occupancy
    helpers.generate_empty_occupancy(dataset_folder, save_folder + "/occupancy")


if __name__ == '__main__':
    package_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_folder = package_dir + '/../data/openstreet/benchmark'
    download_city_crops(dataset_folder)
    create_city_crops_inverted(dataset_folder)
    download_foundations(dataset_folder)
    create_foundations(dataset_folder)
    create_foundations_traversable(dataset_folder)
    create_foundations_inverted(dataset_folder)