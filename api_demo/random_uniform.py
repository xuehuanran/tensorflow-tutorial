# uniform distribution
import tensorflow as tf
import matplotlib.pyplot as plt

x = tf.random_uniform([1000])
session = tf.Session()
x = session.run(x)

plt.plot(x, '.')
plt.show()
