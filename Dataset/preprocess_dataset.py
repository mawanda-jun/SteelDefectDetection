import os
from typing import List, Dict
from config import conf
from PIL import Image
from tqdm import tqdm
import h5py
import numpy as np


def images_in_paths(folder_path: str) -> List[str]:
    """
    Collects paths to all images from one folder and return them as a list
    :param folder_path:
    :return: list of path/to/image
    """
    paths = []
    folder_path = os.path.join(os.getcwd(), folder_path)
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            paths.append(os.path.join(root, file))
    return paths


def convert_grayscale(img_path: str):
    """
    Convert the image in img_path to grayscale. It overwrites existing image
    :param img_path:
    :return:
    """
    img = Image.open(img_path).convert('L')
    os.remove(img_path)
    img.save(img_path, quality=100)


def online_variance(paths):
    """
    Calculate mean and variance in an online way (Welford's algorithm)
    :param paths:
    :return:
    """
    mean = np.zeros((conf.img_h, conf.img_w, conf.dest_channels))
    M2 = np.zeros((conf.img_h, conf.img_w, conf.dest_channels))

    i = 0
    for i, path in tqdm(enumerate(paths), total=len(paths)):
        np_img = np.array(Image.open(path))
        np_img = np.expand_dims(np_img, axis=2)

        delta = np.subtract(np_img, mean)
        mean = np.add(mean, np.divide(delta, (i+1)))
        M2 = np.add(M2, np.multiply(delta, np.subtract(np_img, mean)))

    return mean, np.sqrt(np.divide(M2, i))


def main():
    """
    Converts every image in folder. REMEMBER TO CONVERT IMAGES FROM TEST SET AT RUNTIME
    :return:
    """
    paths = images_in_paths(os.path.join(os.getcwd(), conf.resources, "train_images"))
    # for img_path in tqdm(paths, total=len(paths)):
    #     convert_grayscale(img_path)

    # calculate mean and std of training set
    mean, std = online_variance(paths)
    # save it to a info.h5 file
    with h5py.File(os.path.join(conf.resources, "info.h5"), mode='w') as h5_out:
        h5_out.create_dataset('train_mean', (conf.img_h, conf.img_w, conf.dest_channels), np.float32, data=mean)
        h5_out.create_dataset('train_std', (conf.img_h, conf.img_w, conf.dest_channels), np.float32, data=std)
        h5_out.create_dataset('train_dim', (), np.int32, data=len(paths))


if __name__ == '__main__':
    main()


