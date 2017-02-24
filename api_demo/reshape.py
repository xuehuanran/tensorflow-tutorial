import tensorflow as tf

session = tf.Session()

t = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9])
t_reshape = tf.reshape(t, [3, 3])
print(session.run(t_reshape))

t = tf.constant([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])
t_reshape = tf.reshape(t, [2, 4])
print(session.run(t_reshape))

t = tf.constant(
    [
        [[1, 1, 1], [2, 2, 2]],
        [[3, 3, 3], [4, 4, 4]],
        [[5, 5, 5], [6, 6, 6]]
    ]
)
t_reshape = tf.reshape(t, [-1])
print(session.run(t_reshape))
t_reshape = tf.reshape(t, [2, -1])
print(session.run(t_reshape))
t_reshape = tf.reshape(t, [-1, 9])
print(session.run(t_reshape))
t_reshape = tf.reshape(t, [2, -1, 3])
print(session.run(t_reshape))

t = tf.constant([7])
t_reshape = tf.reshape(t, [])
print(session.run(t_reshape))
