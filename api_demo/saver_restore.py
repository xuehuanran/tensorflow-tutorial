import tensorflow as tf

v1 = tf.Variable([1, 1], name='my_v1')
v2 = tf.Variable([1, 1], name='my_v2')

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, 'tmp/model.ckpt')
    print('Model restored.')
    print(sess.run(v1))
    print(sess.run(v2))

