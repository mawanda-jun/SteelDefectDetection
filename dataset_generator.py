import numpy as np
import h5py
import random
import tensorflow as tf
from tensorflow.python.keras.layers import Lambda
import tensorflow.python.keras.backend as K
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE


class DataGenerator:
    """
    CropsGenerator takes care to load images from disk and convert, crop and serve them as a K.data.Dataset
    """

    def __init__(self, conf, ImageIds_EncodedPixels):
        self.train_images_folder = conf.train_images_folder  # path to train images folder
        self.test_images_folder = conf.test_images_folder  # path to test images folder
        self.resources = conf.resources  # path to the resources folder. It contains useful files regarding the dataset
        self.img_w = conf.img_w
        self.img_h = conf.img_h
        self.img_w_res = conf.img_w_res
        self.img_h_res = conf.img_h_res
        self.crops_w = conf.crops_w  # number of crops to divide image in width

        self.ids_ep = ImageIds_EncodedPixels  # dictionary coming from the digest_train_csv.py method. Actual dataset
        self.train_size = conf.train_size

        self.train_id_ep_dict, self.val_id_ep_dict = self.extract_train_val_datasets()  # retrieve dictionaries of training and validation sets

        self.batch_size = conf.batch_size  # training_batch_size
        self.steps_per_train_epoch = len(list(self.train_id_ep_dict.keys())) // self.batch_size
        self.steps_per_val_epoch = len(list(self.val_id_ep_dict.keys())) // self.batch_size

        self.mean_tensor, self.std_tensor = self.get_stats()  # get stats from dataset info

    def get_stats(self):
        """
        Return mean and std from dataset. It has been memorized. If not mean or std have been saved a KeyError is raised.
        :return:
        """
        with h5py.File(os.path.join(self.resources, 'info.h5'), 'r') as h5:
            mean = h5['train_mean'][:].astype(K.floatx())
            std = h5['train_std'][:].astype(K.floatx())
        return mean, std

    def extract_train_val_datasets(self):
        """
        Extract actual dataset from the grouped file from digest_train_csv.py and transform it to a training/validation sets
        Every label is transformed into a matrix 4-depth, and every layer contains its mask
        :return:
        """
        # random.seed(0)  # give random a seed not to mix train and validation sets in case of resuming
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
            same type, so we transform the int value -1 of "no mask" into a string one.
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
        This function takes care to create a row one pair at a time
        :param pair:
        :return:
        """
        # retrieve idx, length from pair
        # def unstack(x): return x
        # unstack_layer = Lambda(unstack)
        # index, length = unstack_layer(pair)
        # index, length = pair[0], pair[1]

        index, length = tf.unstack(pair, name='unstack_pair')

        def not_empty_row():
            # subtract 1 from index. It start from 1 :(
            idx = index - 1
            # the idea is to create a 3-parts array: the one that goes from 0 to index, the real mask, and the one that
            # goes after the mask until the end of the row
            # def variable_zeros_shape(length): return tf.zeros(length, dtype=tf.uint8)

            # def variable_ones_shape(length): return tf.ones(length, dtype=tf.uint8)

            # before = Lambda(variable_zeros_shape)(idx)
            before = tf.zeros(idx, dtype=tf.uint8)
            mask_line = tf.ones(length, dtype=tf.uint8)
            after_index = (self.img_w * self.img_h) - (length + idx)
            after = tf.zeros(after_index, dtype=tf.uint8)
            row = tf.concat((before, mask_line, after), axis=0)
            # see if all went well
            # tf.compat.v1.debugging.assert_equal(K.shape(row)[0], K.math.multiply(self.img_w, self.img_h))
            return row

        def empty_row():
            # if row has length == 0 there is no need to compute all that above
            # an empty value is returned so memory is kept safe. The reduce operation will be faster
            return tf.constant(0, dtype="uint8")

        return tf.cond(tf.math.equal(length, 0), empty_row, not_empty_row)

    def rle_to_mask(self, rle_input):
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

            # def variable_zeros_shape(shape): return tf.zeros(shape, dtype=tf.uint8)

            return tf.zeros((rows, cols), dtype=tf.uint8)
            # return tf.zeros(shape=(rows, cols), dtype="uint8")

        def with_mask():
            """
            If the mask is present we proceed in creating the effective one.
            RLE is an encoding for masks: the mask has the same dimension of the image. The mask is
            stretched along one axis and becomes a img_w*img_h array. RLE contains the index at which the mask starts
            and the length for how long is. So every mask is a tuple with (index_of_mask, mask) and index_of_mask starts
            from 1.
            With this approach, to parallelize the computation, we compute one index-length at a time and per each index
            we create a img_w*img_h array with all zeros but the indexes with the mask.
            A matrix [img_w*img_h, img_w*img_h] is computed.
            After that, we gather the max value per each column in order to obtain the biggest values per each index
            so we can construct the mask back.
            :return:
            """

            # def rle_string_manager(rle_input):
            #     # divide string into substring containing [index, length, index, length, ...]. A ragged tensor is returned
            #     # since the dimension cannot be fixed for all images
            #     rle = tf.strings.split(input=rle_input, sep=' ', result_type='RaggedTensor', name='ragged_tensor')
            #     # reshape string in order to have pairs of [[index, length], [index, lenght], ...]
            #     rle = tf.reshape(rle, [-1, 2])
            #     # convert string to numbers
            #     rle = tf.strings.to_number(rle, tf.int32, name='convert_str_int32')
            #     return rle
            # since the dimension cannot be fixed for all images
            rle = tf.strings.split(input=rle_input, sep=' ', result_type='RaggedTensor', name='ragged_tensor')
            # reshape string in order to have pairs of [[index, length], [index, lenght], ...]
            rle = tf.reshape(rle, [-1, 2])
            # convert string to numbers
            rle = tf.strings.to_number(rle, tf.int32, name='convert_str_int32')

            # rle = Lambda(rle_string_manager)(rle_input)

            # stacked_masks is a matrix [img_w*img_h, img_w*img_h] that contains a row with a portion of a mask, one
            # for each index
            stacked_masks = tf.map_fn(Lambda(self.create_row), rle, name='create_multiple_images', dtype="uint8")
            # reduce matrix to the original mask dimension in order to retrieve the mask in a "img" shape
            mask = tf.reduce_max(stacked_masks, axis=0)
            # debugging
            # K.compat.v1.debugging.assert_equal(K.shape(mask)[0], K.math.multiply(self.img_h, self.img_w))  # check if the stacked lines are equal to the right dimension
            # reshape mask to the image dimension
            mask = tf.reshape(mask, (cols, rows))
            mask = tf.transpose(mask)
            return mask

        # if there is no mask we only have to return an empty one
        mask = tf.cond(tf.math.not_equal(rle_input, '-1'), with_mask, without_mask)
        return mask

    def process_label(self, x, y):
        # map every mask to the rle_to_mask_func function
        label = tf.map_fn(Lambda(self.rle_to_mask), y, name='stack_rle_strings', dtype="uint8")
        # map_fn function creates a transposed stack of mask channels.
        label = tf.reverse(label, axis=[0])
        label = tf.transpose(label, [1, 2, 0])

        return x, label

    def parse_path(self, path, label):
        """
        Read image from disk and apply a label to it
        :param path: path to one image. This is a K.Tensor and contains a string
        :param label:
        :return:
        """

        # read image from disk
        # def read_img(path): return tf.io.read_file(path)
        img = tf.io.read_file(path)

        # img = Lambda(read_img)(path)
        # decode it as jpeg
        img = tf.image.decode_jpeg(img, channels=1)

        return img, label

    def resize_and_norm(self, x, y):
        x = tf.cast(x, dtype=K.floatx())
        # make the image distant from std deviation of the dataset
        # x = tf.math.subtract(x, self.mean_tensor)
        # x = K.math.divide(x, self.std_tensor)
        x -= self.mean_tensor
        x /= self.std_tensor
        x = tf.image.resize_images(x, (self.img_h_res, self.img_w_res), name='reshape_image')
        y = tf.image.resize_images(y, (self.img_h_res, self.img_w_res), name='reshape_label')

        x = tf.cast(x, K.floatx())
        y = tf.cast(y, "uint8")
        # data augmentation
        # img = K.image.random_flip_left_right(img)
        # img = K.image.random_flip_up_down(img)
        return x, y

    def crop_img_and_serve(self, x, y):
        target_height = self.img_h_res
        target_width = self.img_w_res // self.crops_w

        top_left_w = tf.range(start=0, limit=self.img_w_res - 1, delta=target_width,
                              dtype="int32")

        def crop(w):
            crop_x = tf.image.crop_to_bounding_box(x, 0, w, target_height, target_width)
            crop_y = tf.image.crop_to_bounding_box(y, 0, w, target_height, target_width)
            return crop_x, crop_y

        stacked_crops = tf.map_fn(crop, elems=top_left_w, dtype=(K.floatx(), "uint8"), name="stacked_crops")
        return stacked_crops[0], stacked_crops[1]

    def generate_train_set(self):
        """
        Generates the actual dataset. It uses all the functions defined above to read images from disk and create croppings.
        :param mode: train-val-test
        :return: K.data.Dataset
        """
        parse_path_func = lambda x, y: self.parse_path(x, y)
        process_label_func = lambda x, y: self.process_label(x, y)
        resize_func = lambda x, y: self.resize_and_norm(x, y)
        crops_func = lambda x, y: self.crop_img_and_serve(x, y)
        filter_func = lambda x, y: K.any(y)

        batch_size = self.batch_size

        n_el = len(list(self.train_id_ep_dict.keys()))
        ids = []
        labels = []
        for k, v in self.train_id_ep_dict.items():
            ids.append(os.path.join(self.train_images_folder, k))
            labels.append(v)
        # id_tensor = K.constant(ids, dtype=tf.string, shape=([n_el]))
        # label_tensor = K.constant(labels, dtype=tf.string, shape=(n_el, 4))
        id_tensor = ids
        label_tensor = labels
        return (tf.data.Dataset.from_tensor_slices((id_tensor, label_tensor))
                .shuffle(buffer_size=n_el)
                .map(parse_path_func, num_parallel_calls=AUTOTUNE)
                .map(process_label_func, num_parallel_calls=AUTOTUNE)
                .map(resize_func, num_parallel_calls=AUTOTUNE)
                .map(crops_func, num_parallel_calls=AUTOTUNE)  # create crops of image to enlarge output
                .flat_map(
            lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))  # serve crops as new dataset to flat_map array
                .filter(filter_func)
                .batch(batch_size)  # defined batch_size
                .prefetch(AUTOTUNE)  # number of batches to be prefetch.
                .repeat()  # repeats the dataset when it is finished
                )

    def generate_val_set(self):
        """
        Generates the actual dataset. It uses all the functions defined above to read images from disk and create croppings.
        :return: K.data.Dataset
        """
        parse_path_func = lambda x, y: self.parse_path(x, y)
        process_label_func = lambda x, y: self.process_label(x, y)
        resize_func = lambda x, y: self.resize_and_norm(x, y)
        crops_func = lambda x, y: self.crop_img_and_serve(x, y)
        filter_func = lambda x, y: K.equal(K.any(y), False)

        batch_size = self.batch_size

        n_el = len(list(self.val_id_ep_dict.keys()))
        ids = []
        labels = []
        for k, v in self.val_id_ep_dict.items():
            ids.append(os.path.join(self.train_images_folder, k))
            labels.append(v)
        id_tensor = K.constant(ids, dtype=tf.string, shape=([n_el]))
        label_tensor = K.constant(labels, dtype=tf.string, shape=(n_el, 4))
        return (tf.data.Dataset.from_tensor_slices((id_tensor, label_tensor))
                .shuffle(buffer_size=n_el)
                .map(parse_path_func, num_parallel_calls=AUTOTUNE)
                .map(process_label_func, num_parallel_calls=AUTOTUNE)  # create actual one_crop
                .map(resize_func, num_parallel_calls=AUTOTUNE)  # create actual one_crop
                .map(crops_func, num_parallel_calls=AUTOTUNE)  # create crops of image to enlarge output
                .flat_map(
            lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))  # serve crops as new dataset to flat_map array
                .filter(filter_func)
                .batch(batch_size)  # defined batch_size
                .prefetch(AUTOTUNE)  # number of batches to be prefetch.
                .repeat()  # repeats the dataset when it is finished
                )


# # UNCOMMENT ADDITION AND DIVISION PER MEAN AND STD BEFORE TRY TO SEE IMAGES
if __name__ == '__main__':
    from config import conf
    from Dataset.digest_train_csv import Digestive
    from PIL import Image

    tf.compat.v1.enable_eager_execution()

    # visualize steel image with four classes of faults in seperate columns
    grouped_ids = Digestive(conf).masks_at_least()
    data_reader = DataGenerator(conf, grouped_ids)
    training_set = data_reader.generate_train_set()

    # iter = training_set.make_initializable_iterator()
    # x, labels = iter.get_next()
    # with tf.device('/gpu:0'):
    #     with tf.Session() as sess:
    #         sess.run(iter.initializer)
    #       #  returns a batch of images
    # img, label = sess.run([x, labels])
    for img, label in training_set:
        n_image = 0
        img = img.numpy()
        label = label.numpy()
        img = img[n_image]
        label = label[n_image]
        # identify which layer has mask
        label_index = []
        for i in range(label.shape[-1]):
            label_index.append(label[..., i].any())
        # viz_steel_img_mask(img, label)
        print("img shape: {}".format(img.shape))
        print("label shape: {}".format(label.shape))
        print(label_index)
        img = np.array(img, dtype=K.floatx()).squeeze(axis=-1)
        masks = label
        masks = np.array(masks, dtype=np.uint8)
        img *= 1000
        masks *= 255
        Image.fromarray(img).show()
        Image.fromarray(masks).show()
        break
