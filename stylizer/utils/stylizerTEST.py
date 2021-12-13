from DevAPI.settings import BASE_DIR
import tensorflow as tf
import tensorflow_hub as hub
import os
import numpy as np
import cv2

import time


def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img


# def load_image(path_to_img):
#   max_dim = 512
#   img = tf.io.read_file(path_to_img)
#   img = tf.image.decode_image(img, channels=3)
#   img = tf.image.convert_image_dtype(img, tf.float32)

#   shape = tf.cast(tf.shape(img)[:-1], tf.float32)
#   long_dim = max(shape)
#   scale = max_dim / long_dim

#   new_shape = tf.cast(shape * scale, tf.int32)

#   img = tf.image.resize(img, new_shape)
#   img = img[tf.newaxis, :]
#   return img


# TODO: Load VGG19
# vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

content_layers = ['block5_conv2']
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs*255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(
            inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}


def style_content_loss(outputs):
    style_weight = 1.5
    content_weight = 2
    style_targets = outputs['style']
    content_targets = outputs['origin']
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2)
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2)
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss


def stylizerTEST(uid, path_to_origin, path_to_style):
    # Load pretrained model
    cwd = os.getcwd()
    model = hub.load(
        cwd + '/stylizer/models')

    # TODO: Local test
    # content_image = load_image(
    #     cwd + '/static/images/test/origins/tubingen.jpg')
    # style_image = load_image(
    #     cwd + '/static/images/test/styles/starry-night.jpg')

    load_start_t = time.time()
    # Load image from users
    content_path = tf.keras.utils.get_file(
        cwd + '/static/images/users/origin-image-' + str(uid) + '.jpg', path_to_origin)
    style_path = tf.keras.utils.get_file(
        cwd + '/static/images/users/style-image-' + str(uid) + '.jpg', path_to_style)
    content_image = load_image(content_path)
    style_image = load_image(style_path)
    load_end_t = time.time()
    load_elapse_t = load_end_t - load_start_t

    render_start_t = time.time()
    # Decompose and Reconstruct
    stylized_image = model(tf.constant(content_image),
                           tf.constant(style_image))[0]
    render_end_t = time.time()
    render_elapse_t = render_end_t - render_start_t
    
    # static root is '/static/images/'
    path_to_solved = 'stylized/stylized-image-' + str(uid) + '.jpg'
    path_to_save = cwd + '/static/images/stylized/stylized-image-' + \
        str(uid) + '.jpg'
    # Save
    cv2.imwrite(path_to_save,
                cv2.cvtColor(np.squeeze(stylized_image)*255, cv2.COLOR_BGR2RGB))

    return path_to_solved, load_elapse_t, render_elapse_t
