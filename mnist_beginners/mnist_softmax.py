from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('mnist_data', one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros(10))

y = tf.matmul(x, w) + b

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
