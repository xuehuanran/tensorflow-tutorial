import tensorflow as tf

# define a simple computation graph
input1 = tf.constant([1.0, 2.0, 3.0])
input2 = tf.Variable(tf.random_uniform([3]))
output = tf.add_n([input1, input2])

writer = tf.summary.FileWriter('log', tf.get_default_graph())
writer.close()
