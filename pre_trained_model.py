from MatConvNet import MatConvNet

import os
import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize


class vgg:
    def __init__(self, image, weight_file):

        model = MatConvNet(weight_file)

        matdata = model.loadmat()
        layers_seq, layers, meta = model.decoding(matdata, nkeys=999)

        self.cls = meta['classes']['description']
        self.avg_image = meta['normalization']['averageImage']
        self.image_size = meta['normalization']['imageSize']
        input = tf.image.resize_images(image, (self.image_size[0], self.image_size[1]))

        network = {}
        current = input

        for i, layer_name in enumerate(layers_seq):
            layer_type = layers[layer_name]['type']
            print i, layer_type, layer_name
            curr_layers = layers[layer_name]

            if layer_type == 'conv':

                stride = np.asarray(curr_layers['stride'])

                if stride.size > 1:
                    str1 = stride[0]
                    str2 = stride[1]
                else:
                    str1 = str2 = stride

                if layer_name.startswith('fc'):
                    padding = 'VALID'
                else:
                    padding = 'SAME'

                kernel = tf.constant(curr_layers['weights'][0], dtype=tf.float32, name='weights')
                bias = tf.constant(curr_layers['weights'][1], dtype=tf.float32, name='biases')
                conv = tf.nn.conv2d(
                    current,
                    kernel,
                    strides=(1, str1, str2, 1),
                    padding=padding
                )
                current = tf.nn.bias_add(conv, bias)

            elif layer_type == 'relu':
                current = tf.nn.relu(current)
            elif layer_type == 'pool':
                stride = curr_layers['stride']

                if stride.size > 1:
                    str1 = stride[0]
                    str2 = stride[1]
                else:
                    str1 = str2 = stride

                pool = curr_layers['pool']

                current = tf.nn.max_pool(current,
                                         ksize=[1, pool[0], pool[1], 1],
                                         strides=(1, str1, str2, 1),
                                         padding='SAME',
                                         name='pool1'
                                         )
            elif layer_type == 'lrn':
                current = tf.nn.local_response_normalization(current, depth_radius=2, bias=2.000, alpha=0.0001, beta=0.75)
            elif layer_type == 'softmax':
                current = tf.nn.softmax(tf.reshape(current, [-1, len(self.cls)]))

            #building-up the model layer by layer
            network[layer_name] = current

        self.network = current

def main():

    user_home = os.path.expanduser("~")

    #NO GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    weight_file = user_home + '/models/imagenet-vgg-f.mat'
    img_path = user_home + '/dataset/wine_bottle.jpg'

    img = imread(img_path, mode='RGB')

    image = tf.placeholder(tf.float32, [None, 224, 224, 3])
    model = vgg(image, weight_file)

    #resizing and subtracting the mean img
    img = imresize(img, [224, 224])
    img = img - model.avg_image

    with tf.Session() as sess:

        # Initializing the variables
        init = tf.global_variables_initializer()
        sess.run(init)

        prob = sess.run(model.network, feed_dict={image: [img]})[0]

        #displaying the first five predicted classes
        preds = (np.argsort(prob)[::-1])[0:5]
        for p in preds:
            print p
            print model.cls[p], prob[p]

if __name__ == '__main__':
    main()
