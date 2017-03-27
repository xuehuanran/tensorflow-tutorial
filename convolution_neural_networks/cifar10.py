import tensorflow as tf
import os

FLAGS = tf.app.flags.FLAGS


def download_and_extract():
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)