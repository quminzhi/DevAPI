from DevAPI.settings import BASE_DIR
import tensorflow as tf
import tensorflow_hub as hub
import os
import numpy as np
import cv2


def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img


def stylizer(uid, path_to_origin, path_to_style):
    # Load pretrained model
    cwd = os.getcwd()
    model = hub.load(
        cwd + '/stylizer/models')

    # TODO: Local test
    # content_image = load_image(
    #     cwd + '/static/images/test/origins/tubingen.jpg')
    # style_image = load_image(
    #     cwd + '/static/images/test/styles/starry-night.jpg')

    # Load image from users
    content_path = tf.keras.utils.get_file(
        cwd + '/static/images/users/origin-image-' + str(uid) + '.jpg', path_to_origin)
    style_path = tf.keras.utils.get_file(
        cwd + '/static/images/users/style-image-' + str(uid) + '.jpg', path_to_style)
    content_image = load_image(content_path)
    style_image = load_image(style_path)

    # Decompose and Reconstruct
    stylized_image = model(tf.constant(content_image),
                           tf.constant(style_image))[0]

    # static root is '/static/images/'
    path_to_solved = 'stylized/stylized-image-' + str(uid) + '.jpg'
    path_to_save = cwd + '/static/images/stylized/stylized-image-' + \
        str(uid) + '.jpg'
    # Save
    cv2.imwrite(path_to_save,
                cv2.cvtColor(np.squeeze(stylized_image)*255, cv2.COLOR_BGR2RGB))

    return path_to_solved
