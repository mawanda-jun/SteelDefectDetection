from config import conf
from tqdm import tqdm
import numpy as np
import pandas as pd
import os


def generate_training_dataframe(conf):
    """
    Takes the csv and convert it to a dataframe.
    This dataframe is <grouped_EncodedPixels> and is composed by:
    - the ImageId
    - all the associated masks as a list of tuples.
    This dataframe is accessible via keys, values and items in a loop cycle.
    :param conf:
    :return:
    """
    train_df = pd.read_csv(os.path.join(conf.resources, 'train.csv')).fillna(-1)
    train_df['ImageId'] = train_df["ImageId_ClassId"].apply(lambda x: x.split('_')[0])
    train_df['ClassId'] = train_df["ImageId_ClassId"].apply(lambda x: x.split('_')[1])
    train_df.pop("ImageId_ClassId")
    train_df['ClassId_EncodedPixels'] = train_df.apply(lambda row: (row['ClassId'], row['EncodedPixels']), axis=1)
    grouped_EncodedPixels = train_df.groupby('ImageId')['ClassId_EncodedPixels'].apply(list)
    # # use this code only to select images with more than 1, 2, 3 labels
    # new_dict = {}
    # for k, v in grouped_EncodedPixels.items():
    #     flag = 0
    #     for mask in v:
    #         if -1 in mask:
    #             flag += 1
    #     if flag < 3:
    #         new_dict[k] = v
    # return new_dict
    return grouped_EncodedPixels


# from https://www.kaggle.com/robertkag/rle-to-mask-converter
def rle_to_mask(rle_string, height, width):
    """"
    convert RLE(run length encoding) string to numpy array

    Parameters:
    rleString (str): Description of arg1
    height (int): height of the mask
    width (int): width of the mask

    Returns:
    numpy.array: numpy array of the mask
    """
    rows, cols = height, width
    if rle_string == -1:
        return np.zeros((height, width))
    else:
        rleNumbers = [int(numstring) for numstring in rle_string.split(' ')]
        rlePairs = np.array(rleNumbers).reshape(-1, 2)
        img = np.zeros(rows*cols, dtype=np.uint8)
        for index, length in rlePairs:
            index -= 1
            img[index:index+length] = 255
        img = img.reshape(cols, rows)
        img = img.T
        return img


# Thanks to the authors of: https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
def mask_to_rle(mask):
    """
    Convert a mask into RLE

    Parameters:
    mask (numpy.array): binary mask of numpy array where 1 - mask, 0 - background

    Returns:
    sring: run length encoding
    """
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


if __name__ == '__main__':
    grouped_EncodedPixels = generate_training_dataframe(conf)
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
    new_dict = {}
    keys = list(grouped_EncodedPixels.keys())
    for key in keys:
        new_string = list(zip(*grouped_EncodedPixels[key]))[1]
        print(len(new_string))
        for string in new_string:
            if type(string) == str:
                if "166947 180 167201 182 167456 183 167711 184 167966 185 168221 185" in string:
                    raise ValueError(new_string)


        new_dict[key] = new_string

    for k, v in new_dict.items():
        print("k: {}".format(k))
        print("v: {}".format(v))
        break