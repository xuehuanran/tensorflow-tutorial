import tensorflow as tf

global_step = tf.Variable(initial_value=0, trainable=False)
start_learning_rate = 0.1
learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 100000, 0.96, staircase=True)

session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)

