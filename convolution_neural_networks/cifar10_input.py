import os
import tensorflow as tf


def read_cifar10(filename_queue):
    pass


def distored_inputs(data_dir, batch_size):
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin') for x in range(1, 6)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file ' + f)
    filename_queue = tf.train.string_input_producer(filenames)
