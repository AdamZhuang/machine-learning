import tensorflow as tf
import numpy as np


class Vgg19:
    def __init__(self, x, pretrain_model_path='./vgg19.npy'):
        # load pretrain vgg19 network params
        self.data_dict = np.load(pretrain_model_path,encoding='latin1').item()
        self.net = {}
        self.x = x
        self.build(self.x)
    def build(self, input, clear_data=True):
        """
        load variable from npy to build the VGG
        """
        self.conv1_1 = self.conv_layer(input, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, "conv3_4")
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, "conv4_4")
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")

        self.net['conv1_1'] = self.conv1_1
        self.net['conv1_2'] = self.conv1_2
        self.net['pool1'] = self.pool1
        self.net['conv2_1'] = self.conv2_1
        self.net['conv2_2'] = self.conv2_2
        self.net['pool2'] = self.pool2
        self.net['conv3_1'] = self.conv3_1
        self.net['conv3_2'] = self.conv3_2
        self.net['conv3_3'] = self.conv3_3
        self.net['conv3_4'] = self.conv3_4
        self.net['pool3'] = self.pool3
        self.net['conv4_1'] = self.conv4_1
        self.net['conv4_2'] = self.conv4_2
        self.net['conv4_3'] = self.conv4_3
        self.net['conv4_4'] = self.conv4_4
        self.net['pool4'] = self.pool4
        self.net['conv5_1'] = self.conv5_1

        if clear_data:
            self.data_dict = None

    def get_all_layers(self):
        return self.net

    def max_pool(self, input, name):
        return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, input, name):
        with tf.variable_scope(name):
            filter = self.get_conv_filter(name)
            conv = tf.nn.conv2d(input, filter, [1, 1, 1, 1], padding='SAME')
            conv_biases = self.get_bias(name)
            result = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(result)
            return relu

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")


if __name__ == '__main__':
    model = Vgg19('./vgg19.npy')
    print(1)