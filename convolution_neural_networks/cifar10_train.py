import tensorflow as tf
import cifar10

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', 'cifar10_train')


def main(argv=None):
    cifar10.download_and_extract()


if __name__ == '__main__':
    tf.app.run()
