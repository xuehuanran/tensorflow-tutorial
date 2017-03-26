# normal distribution
import tensorflow as tf
import matplotlib.pyplot as plt

x = tf.random_normal([1000])
session = tf.Session()
x = session.run(x)

plt.plot(x, '.')
plt.show()