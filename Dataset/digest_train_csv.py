from config import conf
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from PIL import Image


class Digestive:
    def __init__(self, configuration):
        """
        Takes the csv and convert it to a dataframe.
        This dataframe is <grouped_EncodedPixels> and is composed by:
        - the ImageId
        - all the associated masks as a list of tuples.
        This dataframe is accessible via keys, values and items in a loop cycle.
        :return:
        """
        train_df = pd.read_csv(os.path.join(configuration.resources, 'train.csv')).fillna(-1)
        train_df['ImageId'] = train_df["ImageId_ClassId"].apply(lambda x: x.split('_')[0])
        train_df['ClassId'] = train_df["ImageId_ClassId"].apply(lambda x: x.split('_')[1])
        train_df.pop("ImageId_ClassId")
        train_df['ClassId_EncodedPixels'] = train_df.apply(lambda row: (row['ClassId'], row['EncodedPixels']), axis=1)
        self.grouped_EncodedPixels = train_df.groupby('ImageId')['ClassId_EncodedPixels'].apply(list)

    def masks_at_least(self, n_masks=1):
        """
        Reads the grouped encoded pixel and keep only those images that have at least <n_masks> masks
        :param n_masks:
        :return:
        """
        new_dict = {}
        for k, v in self.grouped_EncodedPixels.items():
            present_masks = 0
            for mask in v:
                if type(mask[1]) is str:
                    present_masks += 1
            if present_masks > n_masks - 1:
                new_dict[k] = v
        return new_dict

    def keep_masks(self, masks=(0, 1, 2, 3)):
        """
        Reads the grouped encoded pixel and keep only those images that have at least one of the <masks> masks
        :param masks: dict with those images that have the masks we want to keep
        :return:
        """
        masks_only = self.masks_at_least(1)
        new_dict = {}
        for k, v in masks_only.items():
            present_mask = 0
            for n_mask in masks:
                if type(v[n_mask][1]) is str:
                    present_mask += 1
            if present_mask > 0:
                new_dict[k] = v
        return new_dict


def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle2mask(mask_rle, shape=(1600, 256)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


def write_masks_on_disk(masks, folder_path, shape=(256, 1600, 4)):
    os.makedirs(folder_path, exist_ok=False)
    for k, v in tqdm(masks.items()):
        img = np.zeros((shape), dtype=np.uint8)
        out_path = os.path.join(folder_path, k+".png")
        for n_mask, mask in v:
            if mask is not -1:
                img[:, :, int(n_mask)-1] = rle2mask(mask)
        Image.fromarray(img).save(out_path, mode='RGBA')


if __name__ == '__main__':
    only_masks = Digestive(conf).keep_masks([0, 1, 2, 3])
    # print("SANITY CHECK 1.\nChecks if every key in the grouped ids is available inside training images folder")
    # training_img_list = os.listdir(os.path.join(os.getcwd(), "Dataset", "train_images"))
    # found = False
    # for k, v in tqdm(grouped_EncodedPixels.items(), total=len(grouped_EncodedPixels.keys())):
    #     for i, img in enumerate(training_img_list):
    #         if k in img:
    #             found = True
    #             break
    #     if not found:
    #         raise FileNotFoundError("{} has not been found".format(k))
    #     training_img_list.pop(i)
    #     found = False
    #
    # print("SANITY CHECK 2.\nChecks if every image in the training images folder has a label attached.")
    # training_img_list = os.listdir(os.path.join(os.getcwd(), "Dataset", "train_images"))
    # img_ids = list(grouped_EncodedPixels.keys())
    # found = False
    # for img in tqdm(training_img_list, total=len(training_img_list)):
    #     for i, id in enumerate(img_ids):
    #         if img in id:
    #             found = True
    #             break
    #     if not found:
    #         raise FileNotFoundError("{} image has no corrispondent label".format(img))
    #     img_ids.pop(i)
    #     found = False
    # print(list(zip(*grouped_EncodedPixels.items())))
    # for k, v in only_masks.items():
    #     found_mask = False
    #     for mask in v:
    #         if mask is not -1:
    #             found_mask = True
    #             break
    #     if not found_mask:
    #         raise ValueError("There are empty masks")
    write_masks_on_disk(only_masks, os.path.join("Dataset", "masks"))
    print("Number of images with at least one mask: {}".format(len(list(only_masks.keys()))))
