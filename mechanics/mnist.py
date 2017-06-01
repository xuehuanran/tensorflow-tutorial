import tensorflow as tf
import math

NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


def inference(images, hidden1_units, hidden2_units):
    with tf.name_scope('hidden1'):
        weights = tf.Variable(tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                                                  stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))))
        biases = tf.Variable(tf.zeros([hidden1_units]))
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

    with tf.name_scope('hidden2'):
        weights = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units],
                                                  stddev=1.0 / math.sqrt(float(hidden1_units))))
        biases = tf.Variable(tf.zeros([hidden2_units]))
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(tf.truncated_normal([hidden2_units, NUM_CLASSES],
                                                  stddev=1.0 / math.sqrt(float(hidden2_units))))
        biases = tf.Variable(tf.zeros([NUM_CLASSES]))
        logits = tf.matmul(hidden2, weights) + biases

    return logits


def loss(logits, labels):
    pass


def training(loss, learning_rate):
    pass


def evaluation(logits, labels):
    pass
