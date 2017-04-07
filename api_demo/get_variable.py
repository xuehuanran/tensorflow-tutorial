import tensorflow as tf
import matplotlib.pyplot as plt

filter_weight = tf.get_variable('weights', [5, 5, 3, 16], initializer=tf.truncated_normal_initializer(stddev=0.1))

session = tf.Session()

initializer = tf.global_variables_initializer()
session.run(initializer)

filter_weight_value = session.run(filter_weight)

plt.plot(filter_weight_value.reshape([1200]))
plt.show()
