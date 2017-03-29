import tensorflow as tf
import cifar10


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', 'cifar10_train', """Path to store the train log.""")


def train():
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()



def main(argv=None):
    cifar10.download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
