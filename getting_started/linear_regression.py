import tensorflow as tf

w = tf.Variable(0.3)
b = tf.Variable(-0.3)

x = tf.placeholder(tf.float32)
linear_model = tf.multiply(w, x) + b
y = tf.placeholder(tf.float32)

loss = tf.reduce_sum(tf.square(linear_model - y))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(1000):
    sess.run(train, feed_dict={x: x_train, y: y_train})

current_w, current_b, current_loss = sess.run([w, b, loss], feed_dict={x: x_train, y: y_train})
print("w: ", current_w, "  b: ", current_b, " loss: ", current_loss)
