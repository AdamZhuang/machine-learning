import tensorflow as tf
import numpy as np
import vgg19
import cv2

CONTENT_LAYERS = ['conv4_2']
STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

alpha = 0.01
beta = 1
lr = 10

class StyleTransfer(object):
    def __init__(self, content_image_path, style_image_path):
        content = self.preprocess(cv2.imread(content_image_path))
        style = self.preprocess(cv2.imread(style_image_path))

        self.content = np.reshape(content,(1,)+content.shape)
        self.style = np.reshape(style, (1,) + style.shape)

        # store feature
        self.content_feature = {}
        self.style_feature = {}

        # build graph
        self.build_graph()


    def build_graph(self):
        # what will be train
        init_image = np.random.normal(size=self.content.shape, scale=np.std(self.content))
        self.x = tf.Variable(init_image,trainable=True,dtype=tf.float32)

        # get content features
        with tf.Graph().as_default():
            self.input_content = tf.placeholder(tf.float32, shape=self.content.shape)
            net = vgg19.Vgg19(self.input_content).get_all_layers()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for layer in CONTENT_LAYERS:
                    featuremap = sess.run(net[layer],feed_dict={self.input_content:self.content})
                    self.content_feature[layer] = featuremap

        # get content features
        with tf.Graph().as_default():
            self.input_style = tf.placeholder(tf.float32, shape=self.style.shape)
            net = vgg19.Vgg19(self.input_style).get_all_layers()
            with tf.Session() as sess:
                for layer in STYLE_LAYERS:
                    featuremap = net[layer]
                    gram = self.gram_matrix(featuremap)
                    self.style_feature[layer] = sess.run(gram, feed_dict={self.input_style:self.style})

        # get gen_image net
        self.gen_image_net = vgg19.Vgg19(self.x).get_all_layers()


        # loss definition
        self.l_content = 0
        self.l_style = 0
        self.l_total = 0

        # compute loss
        for layer in CONTENT_LAYERS:
            w = 1 / len(CONTENT_LAYERS)
            self.l_content += w * tf.reduce_sum(tf.pow((self.gen_image_net[layer] - self.content_feature[layer]), 2)) / 2


        for layer in STYLE_LAYERS:
            _, height, width, dim = self.gen_image_net[layer].get_shape()
            N = height.value * width.value
            M = dim.value
            w = 1 / len(STYLE_LAYERS)
            gram_style = self.style_feature[layer]
            gram_gen = self.gram_matrix(self.gen_image_net[layer])

            self.l_style += w * (1. / (4 * N ** 2 * M ** 2)) * tf.reduce_sum(tf.pow((gram_gen-gram_style), 2))

        self.l_total = alpha * self.l_content + beta * self.l_style

        self.train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.l_total)

    def update(self, iter_num):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(iter_num):
                sess.run(self.train_step)
                

                if(i % 5 == 0):
                    l_total, l_content, l_style = sess.run([self.l_total,self.l_content,self.l_style])

                    print('iter : %4d, ' % i, 'L_total : %g, L_content : %g, L_style : %g' % (l_total, l_content, l_style))

                    gen_image = sess.run(self.x)
                    image = np.reshape(gen_image,gen_image.shape[1:])
                    image = self.unprocess(image)
                    cv2.imwrite('./gen/'+str(i)+'.jpg', image)


    def gram_matrix(self, tensor):
        shape = tensor.get_shape()
        num_channels = int(shape[3])
        matrix = tf.reshape(tensor, shape=[-1, num_channels])
        gram = tf.matmul(tf.transpose(matrix), matrix)
        return gram


    def preprocess(self, image):
        return image - [123.68, 116.779, 103.939]

    def unprocess(self, image):
        return image + [123.68, 116.779, 103.939]

if __name__ == '__main__':
    print('construct model...')
    model = StyleTransfer('./images/11.jpg', './images/22.jpg')
    print('training...')
    model.update(1000)
