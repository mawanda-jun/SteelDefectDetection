import numpy as np
import h5py
import random
import tensorflow as tf
import os
from typing import List

AUTOTUNE = tf.data.experimental.AUTOTUNE


# def images_in_paths(folder_path: str) -> List[str]:
#     """
#     Collects paths to all images from one folder and return them as a list
#     :param folder_path:
#     :return: list of path/to/image
#     """
#     paths = []
#     folder_path = os.path.join(os.getcwd(), folder_path)
#     for root, dirs, files in os.walk(folder_path):
#         for file in files:
#             paths.append(os.path.join(root, file))
#     return paths
#
#
# class ImageGenerator:
#     """
#     ImageGenerator takes care to serve all path/to/images in the right way and to serve the dataset info about mean,
#     standard deviation and the number of items in folder.
#     """
#
#     def __init__(self, data_path):
#         """
#         Open file once. The object ImageGenerator is kept alive for all the training (and evaluation). So lists and file
#         are not loaded to disk yet, waiting for call function.
#         :param data_path: path/to/data folder, in which there are train, val, test folders with images and the file 'info.h5' with
#         std/mean and number of items in dataset
#         """
#         self.data_path = data_path
#         self.h5f = h5py.File(os.path.join(data_path, 'info.h5'),
#                              'r')  # it will be closed when the context will be terminated
#
#     def __call__(self, dataset_type, num_classes, *args, **kwargs):
#         """
#         Instance is called with different dataset_type
#         :param dataset_type:
#         :param args:
#         :param kwargs:
#         :return:
#         """
#         paths = images_in_paths(os.path.join(self.data_path, dataset_type))
#         labels = [random.randrange(num_classes) for _ in paths]
#         return paths, labels
#

class DataGenerator:
    """
    CropsGenerator takes care to load images from disk and convert, crop and serve them as a tf.data.Dataset
    """

    def __init__(self, conf, ImageIds_EncodedPixels):
        self.train_images_folder = conf.train_images_folder  # path to train images folder
        self.test_images_folder = conf.test_images_folder  # path to test images folder
        self.resources = conf.resources  # path to the resources folder. It contains useful files regarding the dataset
        self.img_w = conf.img_w
        self.img_h = conf.img_h

        self.ids_ep = ImageIds_EncodedPixels  # dictionary coming from the digest_train_csv.py method. Actual dataset
        self.train_size = conf.train_size

        self.train_id_ep_dict, self.val_id_ep_dict = self.extract_train_val_datasets()  # retrieve dictionaries of training and validation sets

        self.batch_size = conf.batch_size  # training_batch_size
        self.mean_tensor, self.std_tensor = self.get_stats()  # get stats from dataset info

    def get_stats(self):
        """
        Return mean and std from dataset. It has been memorized. If not mean or std have been saved a KeyError is raised.
        :return:
        """
        with h5py.File(os.path.join(self.resources, 'info.h5'), 'r') as h5:
            mean = h5['train_mean'][:].astype(np.float32)
            std = h5['train_std'][:].astype(np.float32)
        return mean, std

    def extract_train_val_datasets(self):
        """
        Extract actual dataset from the grouped file from digest_train_csv.py and transform it to a training/validation sets
        Every label is transformed into a matrix 4-depth, and every layer contains its mask
        :return:
        """
        random.seed(3)  # give random a seed not to mix train and validation sets in case of resuming
        keys = list(self.ids_ep.keys())
        random.shuffle(keys)
        train_examples = round(len(keys) * self.train_size)
        train_keys = keys[0:train_examples]
        val_keys = keys[train_examples:-1]
        train_id_ep = {}
        val_id_ep = {}

        def convert_ep(k):
            """
            Encode labels in strings to maintain compatibility with tensorflow API. x and y must have
            same type.
            :param k:
            :return:
            """
            encoded_pixels = self.ids_ep[k]
            array_of_ep = list((list(zip(*encoded_pixels))[1]))
            for i, val in enumerate(array_of_ep):
                if isinstance(val, int):
                    array_of_ep[i] = '-1'
            return array_of_ep

        for key in train_keys:
            train_id_ep[key] = convert_ep(key)
        for key in val_keys:
            val_id_ep[key] = convert_ep(key)

        return train_id_ep, val_id_ep

    def create_row(self, pair):
        """
        Takes care to create the row of the
        :param pair:
        :return:
        """
        # retrieve idx, length from pair
        index, length = tf.unstack(pair, name='unstack_pair')

        def not_empty_row():
            # subtract 1 from index. It start from 1 :(
            idx = tf.math.subtract(index, 1)
            # the idea is to create a 3-parts array: the one that goes from 0 to index, the real mask, and the one that
            # goes after the mask until the end of the row
            before = tf.zeros([idx], dtype=tf.uint8, name='before_ep')
            mod = tf.cast(tf.fill([length], 255, name='ep'), dtype=tf.uint8)
            mod = tf.cast(mod, dtype=tf.uint8)
            after_index = tf.math.subtract(tf.math.multiply(self.img_w, self.img_h), tf.add(length, idx))
            after = tf.zeros([after_index], dtype=tf.uint8, name='after_ep')
            row = tf.concat((before, mod, after), axis=tf.constant(0), name='concat_row')
            # see if all went well
            tf.debugging.assert_equal(tf.shape(row)[0], tf.math.multiply(self.img_w, self.img_h))
            return row

        def empty_row():
            # if row has length == 0 there is no need to compute all that above
            return tf.zeros([tf.math.multiply(self.img_w, self.img_h)], dtype=tf.uint8)

        return tf.cond(tf.math.equal(length, 0), empty_row, not_empty_row, name="row_empty_or_not")

    def rle_to_mask(self, rle_input: tf.Tensor):
        """"
        convert RLE(run length encoding) string to numpy array

        Returns:
        numpy.array: numpy array of the mask
        """
        rows, cols = self.img_h, self.img_w

        def without_mask():
            """
            If layer does not contain any mask it returns an empty one
            :return:
            """
            return tf.zeros(shape=(rows, cols), dtype=tf.uint8)

        @tf.contrib.eager.defun
        def with_mask():
            """
            If the mask is present we proceed in creating the effective one.
            RLE is an encoding for masks: the mask has the same dimension of the image. The mask is
            stretched along one axis and becomes a img_w*img_h array. RLE contains the index at which the mask starts
            and the length for how long is.
            With this approach, to parallelize the computation, we compute one index-length at a time and per each index
            we create a img_w*img_h array with all zeros but the indexes with the mask.
            A matrix [img_w*img_h, img_w*img_h] is computed.
            After that, we gather the max value per each column in order to obtain the biggest values per each index
            so we can construct the mask back. This is parallelism proof.
            :return:
            """
            # divide string into substring containing [index, length, index, length, ...]. A ragged tensor is returned
            # since the dimension cannot be fixed for all images
            rle = tf.strings.split(input=rle_input, sep=' ', result_type='RaggedTensor', name='ragged_tensor')
            # reshape string in order to have pairs of [[index, length], [index, lenght], ...]
            rle = tf.manip.reshape(rle, (-1, 2), name='reshape_string')
            # convert string to numbers
            rle = tf.strings.to_number(rle, tf.int32, name='convert_str_int32')

            # this function takes care to create a row one pair at a time
            create_row_func = lambda pair: self.create_row(pair)

            # stacked_masks is a matrix [img_w*img_h, img_w*img_h] that contains a row with a portion of a mask, one
            # for each index
            stacked_masks = tf.map_fn(create_row_func, rle, name='create_multiple_images', dtype=tf.uint8)
            # we transpose the stacked masks in order to have the values we want to reduce in the same array
            stacked_masks = tf.transpose(stacked_masks, name='transpose_lines')
            # reduce matrix to the original mask dimension in order to retrieve the original mask
            mask = tf.reduce_max(stacked_masks, axis=1, name='find_max_and_create_one_line')
            # check if all went well
            tf.debugging.assert_equal(tf.shape(mask)[0], tf.math.multiply(self.img_h, self.img_w))  # check if the stacked lines are equal to the right dimension
            # reshape mask to the image dimension
            mask = tf.reshape(mask, (cols, rows), name='last_reshape')
            mask = tf.transpose(mask, name='last_transpose')
            return mask
        # if there is no mask we only have to return an empty one
        mask = tf.cond(tf.math.equal(rle_input, tf.constant('-1')), without_mask, with_mask, name='condition_for_mask')
        return mask

    @tf.contrib.eager.defun  # takes care to parallelize map_fn since in eager_execution is disabled
    def process_label(self, x: tf.Tensor, y: tf.Tensor):
        # y is made up of 4 stacked masks. We want to process one at a time
        rle_to_mask_func = lambda rle_string: self.rle_to_mask(rle_string)
        # map every mask to the rle_to_mask_func function
        label = tf.map_fn(rle_to_mask_func, y, name='stack_rle_strings', dtype=tf.uint8)
        return x, label

    def parse_path(self, path: tf.Tensor, label: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        """
        Read image from disk and apply a label to it
        :param path: path to one image. This is a tf.Tensor and contains a string
        :param label:
        :return:
        """
        # read image from disk
        img = tf.io.read_file(tf.strings.join((self.train_images_folder, path), separator='/'))
        # decode it as jpeg
        img = tf.image.decode_jpeg(img, channels=1)
        # data augmentation
        # img = tf.image.random_flip_left_right(img)
        # img = tf.image.random_flip_up_down(img)
        # cast to tensor with type tf.float16
        img = tf.cast(img, dtype=tf.float16)

        # make the image distant from std deviation of the dataset
        # img = tf.math.subtract(img, self.mean_tensor)
        # img = tf.math.divide(img, self.std_tensor)

        return img, label

    def generate_train_set(self):
        """
        Generates the actual dataset. It uses all the functions defined above to read images from disk and create croppings.
        :param mode: train-val-test
        :return: tf.data.Dataset
        """
        parse_path_func = lambda x, y: self.parse_path(x, y)
        process_label_func = lambda x, y: self.process_label(x, y)

        batch_size = self.batch_size

        n_el = len(list(self.train_id_ep_dict.keys()))
        ids = []
        labels = []
        for k, v in self.train_id_ep_dict.items():
            ids.append(k)
            labels.append(v)
        id_tensor = tf.constant(ids, dtype=tf.string, shape=([n_el]))
        label_tensor = tf.constant(labels, dtype=tf.string, shape=(n_el, 4))
        return (tf.data.Dataset.from_tensor_slices((id_tensor, label_tensor))
                .shuffle(buffer_size=n_el)
                .map(parse_path_func, num_parallel_calls=AUTOTUNE)
                .map(process_label_func, num_parallel_calls=AUTOTUNE)  # create actual one_crop
                .batch(batch_size)  # defined batch_size
                .prefetch(AUTOTUNE)  # number of batches to be prefetch.
                .repeat()  # repeats the dataset when it is finished
                )

    # def generate_val_set(self):
    #     """
    #     Generates the actual dataset. It uses all the functions defined above to read images from disk and create croppings.
    #     :param mode: train-val-test
    #     :return: tf.data.Dataset
    #     """
    #     parse_path_func = lambda x, y: self.parse_path(x, y)
    #     create_croppings_func = lambda x, y, z: self.rle_to_mask(x, y, z)
    #
    #     batch_size = self.val_batch_size
    #     n_el = self.num_val_batch
    #
    #     return (tf.data.Dataset.from_tensor_slices(self.img_generator('val', self.numClasses))
    #             .shuffle(buffer_size=n_el * batch_size)
    #             .map(parse_path_func, num_parallel_calls=AUTOTUNE)
    #             .map(create_croppings_func, num_parallel_calls=AUTOTUNE)  # create actual one_crop
    #             .batch(batch_size)  # defined batch_size
    #             .prefetch(AUTOTUNE)  # number of batches to be prefetch.
    #             .repeat()  # repeats the dataset when it is finished
    #             )
    #
    # def generate_test_set(self):
    #     """
    #     Generates the actual dataset. It uses all the functions defined above to read images from disk and create croppings.
    #     :param mode: train-val-test
    #     :return: tf.data.Dataset
    #     """
    #     parse_path_func = lambda x, y: self.parse_path(x, y)
    #     create_croppings_func = lambda x, y, z: self.rle_to_mask(x, y, z)
    #
    #     batch_size = self.batchSize
    #     n_el = self.num_test_batch
    #
    #     return (tf.data.Dataset.from_tensor_slices(self.img_generator('test', self.numClasses))
    #             .shuffle(buffer_size=n_el * batch_size)
    #             .map(parse_path_func, num_parallel_calls=AUTOTUNE)
    #             .map(create_croppings_func, num_parallel_calls=AUTOTUNE)  # create actual one_crop
    #             .batch(batch_size)  # defined batch_size
    #             .prefetch(AUTOTUNE)  # number of batches to be prefetch.
    #             .repeat()  # repeats the dataset when it is finished
    #             )


# # UNCOMMENT ADDITION AND DIVISION PER MEAN AND STD BEFORE TRY TO SEE IMAGES
if __name__ == '__main__':
    from config import conf
    import cv2
    from matplotlib import pyplot as plt
    from Dataset.digest_train_csv import generate_training_dataframe
    from PIL import Image
    # visualize steel image with four classes of faults in seperate columns

    def viz_steel_img_mask(img, masks):
        img = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(nrows=1, ncols=4, sharey='all', figsize=(20, 10))
        cmaps = ["Reds", "Blues", "Greens", "Purples"]
        for idx, mask in enumerate(masks):
            ax[idx].imshow(img)
            ax[idx].imshow(mask, alpha=0.3, cmap=cmaps[idx])

    grouped_ids = generate_training_dataframe(conf)

    #
    #     os.chdir(os.pardir)
    #     with h5py.File(os.path.join('Dataset', conf.resources, conf.hammingFileName + str(conf.hammingSetSize) + '.h5'),
    #                    'r') as h5f:
    #         HammingSet = np.array(h5f['max_hamming_set'])
    #
    data_reader = DataGenerator(conf, grouped_ids)
    training_set = data_reader.generate_train_set()

#
    iter = training_set.make_initializable_iterator()
    x, labels = iter.get_next()
#
    with tf.Session() as sess:
        sess.run(iter.initializer)
        # returns a batch of images
        img, label = sess.run([x, labels])
        n_image = 1
        img = img[n_image]
        label = label[n_image]
        img = np.array(img, dtype=np.uint8)
        img = np.squeeze(img)
        label = np.moveaxis(np.array(label, dtype=np.uint8), 0, -1)
        print("img shape: {}".format(img.shape))
        print("label shape: {}".format(label.shape))
        Image.fromarray(img).show()
        Image.fromarray(label).show()
        # viz_steel_img_mask(img, label)

#         # select only one (choose which in [0, batchSize)
#         n_image = 4
#         image = np.array(tiles[n_image], dtype=np.float32)
#         first_label = np.array(labels[n_image])
#         # from one_hot to number
#         lbl = np.where(first_label == np.amax(first_label))[0][0]
#
#         # create complete image with pieces (if label is correct then also will be image)
#         complete = np.zeros((192, 192, 3))
#         tile_size = data_reader.tileSize
#         for i, v in enumerate(data_reader.maxHammingSet[lbl]):
#             row = int(v / 3)
#             col = v % 3
#             y_start = row * tile_size
#             x_start = col * tile_size
#             complete[y_start:y_start + tile_size, x_start:x_start + tile_size] = image[:, :, :, i]
#
        # print(np.array(img).shape)
