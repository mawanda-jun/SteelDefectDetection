import tensorflow as tf
import os

flags = tf.flags

# to keep compatibility with CLI arg
flags.DEFINE_string('task', 'jps', 'what the hell')
flags.DEFINE_integer('batchSize', 100, 'training batch size')

conf = tf.flags.FLAGS
