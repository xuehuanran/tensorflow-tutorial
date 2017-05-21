import tensorflow as tf
import matplotlib.pyplot as plt

img_raw_data = tf.gfile.FastGFile('../dataset/cat.jpg', 'rb').read()
session = tf.Session()

img_data = tf.image.decode_jpeg(img_raw_data)
img_data = tf.image.convert_image_dtype(img_data, tf.float32)

bilinear = tf.image.resize_images(img_data, [300, 300], method=0)
nearset_neighbor = tf.image.resize_images(img_data, [300, 300], method=1)
bicubic = tf.image.resize_images(img_data, [300, 300], method=2)
area = tf.image.resize_images(img_data, [300, 300], method=3)

plt.subplot(221)
plt.imshow(session.run(bilinear))
plt.title('bilinear')

plt.subplot(222)
plt.imshow(session.run(nearset_neighbor))
plt.title('nearset_neighbor')

plt.subplot(223)
plt.imshow(session.run(bicubic))
plt.title('bicubic')

plt.subplot(224)
plt.imshow(session.run(area))
plt.title('area')

plt.show()
