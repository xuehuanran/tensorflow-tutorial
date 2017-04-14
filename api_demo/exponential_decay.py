import tensorflow as tf

global_step = tf.Variable(initial_value=1, trainable=False, dtype=tf.float32)
start_learning_rate = 1.0
mu = 0.9
exp_mu = tf.train.exponential_decay(1.0, global_step, 1.0, mu)
learning_rate = tf.divide(start_learning_rate, (1 - exp_mu))

x = tf.Variable(tf.constant([0], dtype=tf.float32))
loss = tf.pow(x, 2.0)

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)

for i in range(10):
    print('t = ', session.run(global_step))
    print("exp_mu = ", session.run(exp_mu))
    print('learning_rate = ', session.run(learning_rate))
    session.run(train_step)
