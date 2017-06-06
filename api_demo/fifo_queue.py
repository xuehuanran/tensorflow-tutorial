import tensorflow as tf

q = tf.FIFOQueue(2, tf.int32)

init = q.enqueue_many(tf.constant([0, 10]))

x = q.dequeue()

y = x + 1

q_inc = q.enqueue([y])

with tf.Session() as session:
    init.run()
    for i in range(5):
        v, _ = session.run([x, q_inc])
        print(v)
