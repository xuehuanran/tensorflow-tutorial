import tensorflow as tf

image_raw_data = tf.gfile.GFile('../dataset/cat.jpg', 'rb').read()


with tf.Session() as session:
    img_data = tf.image.decode_jpeg(image_raw_data)
    encoded_image = tf.image.encode_jpeg(img_data)
    print(encoded_image.eval())
    with tf.gfile.GFile('../dataset/cat_copy.jpg', 'wb') as f:
        f.write(encoded_image.eval())
