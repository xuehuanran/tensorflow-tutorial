import tensorflow as tf

x = [[1, 4], [2, 5], [3, 6]]

session = tf.InteractiveSession()

print(tf.transpose(x, [0, 1]).eval())

x = [
    [[1, 4], [2, 5], [3, 6]],
    [[7, 10], [8, 11], [9, 12]]
]

print(tf.transpose(x).eval())  # x001 = x100 = 4, x110 = x011, x021 = x120 = 6
