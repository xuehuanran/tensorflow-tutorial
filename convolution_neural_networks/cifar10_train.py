import tensorflow as tf
import cifar10


def main(argv=None):
    cifar10.download_and_extract()


if __name__ == '__main__':
    tf.app.run()
