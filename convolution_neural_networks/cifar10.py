import tensorflow as tf
import os
import sys
from urllib.request import urlretrieve
import tarfile

# FLAGS is a global defined in flags.py
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 128, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', 'cifar10_data', """Path to the CIFAR-10 data directory.""")

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def download_and_extract():
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progess(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100))
            sys.stdout.flush()
        filepath, _ = urlretrieve(DATA_URL, filepath, _progess)
        stat_info = os.stat(filepath)
        print("Successfully downloaded", filename, stat_info.st_size, 'bytes.')
    extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)