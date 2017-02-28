import tensorflow as tf

input_value = tf.constant(0, dtype=tf.float32, shape=[100, 227, 227, 3])
w = tf.Variable(tf.zeros([5, 5, 3, 32]))

output_value_same = tf.nn.conv2d(input_value, w, strides=[1, 2, 3, 1], padding='SAME')
output_value_valid = tf.nn.conv2d(input_value, w, strides=[1, 2, 3, 1], padding='VALID')

print("same padding algorithm:", output_value_same)
print("valid padding algorithm:", output_value_valid)
