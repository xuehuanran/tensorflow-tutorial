import tensorflow as tf

v1 = tf.Variable([1, 2], name='my_v1')
v2 = tf.Variable([3, 4], name='my_v2')

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess, 'tmp/model.ckpt')
    print('Model saved in the file %s' % save_path)
