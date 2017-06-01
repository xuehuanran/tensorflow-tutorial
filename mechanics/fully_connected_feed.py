import argparse
import tensorflow as tf
import sys
from tensorflow.examples.tutorials.mnist import input_data
import mnist

FLAGS = None


def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(tf.float32, [batch_size, mnist.IMAGE_PIXELS])
    labels_placeholder = tf.placeholder(tf.int32, batch_size)
    return images_placeholder, labels_placeholder


def run_training():
    data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)
    with tf.Graph().as_default():
        images_placeholder, lables_placeholder = placeholder_inputs(FLAGS.batch_size)


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.01,
                        help='Initialing learning rate.')
    parser.add_argument('--max_steps',
                        type=int,
                        default=2000,
                        help='Number of steps to run trainer.')
    parser.add_argument('--hidden1',
                        type=int,
                        default=128,
                        help='Number of units in hidden layer 1.')
    parser.add_argument('--hidden2',
                        type=int,
                        default=32,
                        help='Number of units in hidden layer 2.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=100,
                        help='Batch size. Must divide evenly into the dataset size.')
    parser.add_argument('--input_data_dir',
                        type=str,
                        default='../dataset/mnist',
                        help='Directory to put the input data.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put the log data.')
    parser.add_argument('--fake_data',
                        default=False,
                        help='If true, use fake data for unit testing.',
                        action='store_true')

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
