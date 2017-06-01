import tensorflow as tf
import cifar10


def main(_):
    cifar10.maybe_download_and_extract()


if __name__ == '__main__':
    tf.app.run()
