import tensorflow as tf
import os

flags = tf.flags

# configuration for competition

# Directories
flags.DEFINE_string('resources', os.path.join(os.getcwd(), "Dataset"), 'path to dataset and other resources')
flags.DEFINE_string('train_images_folder', os.path.join(os.getcwd(), "Dataset", "train_images"), 'path to training images')
flags.DEFINE_string('test_images_folder', os.path.join(os.getcwd(), "Dataset", "test_images"), 'path to test images')


flags.DEFINE_string("results", os.path.join(os.getcwd(), "Results"), "path/to/results")
flags.DEFINE_string("run_name", "run0", "name of the current run")
flags.DEFINE_integer("reload_step", 0, "step from which start the training. In case of resuming training we have to modify this parameter")

# network configuration
flags.DEFINE_integer("batch_size", 1, "batch size")
flags.DEFINE_float("learning_rate", 0.0001, "learning rate")
flags.DEFINE_integer("epochs", 30, "number of epochs")

# dataset configuration
flags.DEFINE_integer("img_w", 1600, "original width")
flags.DEFINE_integer("img_h", 256, "original height")
flags.DEFINE_integer("channels", 3, "original channels")
flags.DEFINE_integer("img_w_res", 256, "resized width")
flags.DEFINE_integer("img_h_res", 32, "resized height")

flags.DEFINE_integer("dest_channels", 1, "channels after grayscale")

flags.DEFINE_float("train_size", 0.7, "percentile of training dataset")

conf = tf.flags.FLAGS
