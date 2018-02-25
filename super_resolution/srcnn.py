import tensorflow as tf
import numpy as np
import h5py
import cv2


learning_rate = 1e-3
batch_size = 30
epoch = 25
scale = 1/3

class SRCNN(object):
    def __init__(self, sess, dataset_path, epoch, batch_size, scale):
        self.sess = sess
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.epoch = epoch
        self.scale = scale
        self.data = tf.placeholder(tf.float32, [None, None, None, 3], name='images')
        self.labels = tf.placeholder(tf.float32, [None, None, None, 3], name='labels')
        self.build_graph()


    def build_graph(self):
        # spilt to B, G, R
        B, G, R = tf.split(self.data, 3, axis=3)
        # param definition
        self.w1_b = tf.Variable(tf.random_normal([9, 9, 1, 64], stddev=1e-3),  name='w1_b')
        self.w2_b = tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3), name='w2_b')
        self.w3_b = tf.Variable(tf.random_normal([5, 5, 32, 1], stddev=1e-3),  name='w3_b')
        self.w1_g = tf.Variable(tf.random_normal([9, 9, 1, 64], stddev=1e-3),  name='w1_g')
        self.w2_g = tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3), name='w2_g')
        self.w3_g = tf.Variable(tf.random_normal([5, 5, 32, 1], stddev=1e-3),  name='w3_g')
        self.w1_r = tf.Variable(tf.random_normal([9, 9, 1, 64], stddev=1e-3),  name='w1_r')
        self.w2_r = tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3), name='w2_r')
        self.w3_r = tf.Variable(tf.random_normal([5, 5, 32, 1], stddev=1e-3),  name='w3_r')
        self.b1_b = tf.Variable(tf.zeros([64]), name='b1_b')
        self.b2_b = tf.Variable(tf.zeros([32]), name='b2_b')
        self.b3_b = tf.Variable(tf.zeros([1]),  name='b3_b')
        self.b1_g = tf.Variable(tf.zeros([64]), name='b1_g')
        self.b2_g = tf.Variable(tf.zeros([32]), name='b2_g')
        self.b3_g = tf.Variable(tf.zeros([1]),  name='b3_g')
        self.b1_r = tf.Variable(tf.zeros([64]), name='b1_r')
        self.b2_r = tf.Variable(tf.zeros([32]), name='b2_r')
        self.b3_r = tf.Variable(tf.zeros([1]),  name='b3_r')
        # conv layer
        self.conv1_b = tf.nn.relu(self.conv2d(B, self.w1_b) + self.b1_b)
        self.conv2_b = tf.nn.relu(self.conv2d(self.conv1_b, self.w2_b) + self.b2_b)
        self.conv3_b = self.conv2d(self.conv2_b, self.w3_b) + self.b3_b

        self.conv1_g = tf.nn.relu(self.conv2d(G, self.w1_g) + self.b1_g)
        self.conv2_g = tf.nn.relu(self.conv2d(self.conv1_g, self.w2_g) + self.b2_g)
        self.conv3_g = self.conv2d(self.conv2_g, self.w3_g) + self.b3_g

        self.conv1_r = tf.nn.relu(self.conv2d(R, self.w1_r) + self.b1_r)
        self.conv2_r = tf.nn.relu(self.conv2d(self.conv1_r, self.w2_r) + self.b2_r)
        self.conv3_r = self.conv2d(self.conv2_r, self.w3_r) + self.b3_r
        # output
        self.output = tf.concat((self.conv3_b, self.conv3_g, self.conv3_r),axis=3)
        # loss
        self.loss = tf.reduce_mean(tf.square(self.labels - self.output))
        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        # init
        self.sess.run(tf.global_variables_initializer())



    def train(self):
        data = self.read_data(self.dataset_path)
        train_data, train_labels = self.process_origin_data(data, self.scale)
        batch_data, batch_labels = self.get_batch(train_data, train_labels, self.batch_size)
        for i in range(self.epoch):
            for j in range(len(batch_data)):
                one_batch_data, one_batch_labels = batch_data[j], batch_labels[j]
                self.sess.run(self.optimizer, feed_dict={self.data:one_batch_data, self.labels:one_batch_labels})
                print('epoch:', str(i), 'batch:', str(j))

                if( j%5==0 or j+1==len(batch_data)):
                    print('loss:', self.sess.run(self.loss, feed_dict={self.data:one_batch_data, self.labels:one_batch_labels}))

        saver = tf.train.Saver()
        saver.save(self.sess,'./model/model_data.ckpt')


    def load_model(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, './model/model_data.ckpt')
        print('model loaded.')


    def predict(self, low_resolution_image_path, scale=1):
        image = cv2.imread(low_resolution_image_path)
        if(scale != 1):
            width, height, channel = image.shape
            image = cv2.resize(image,(int(scale * width), int(scale * height)))

        # cv2.imshow('1',image)
        # cv2.waitKey(0)
        # 转化模式
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        # cv2.imshow('1', image)
        # cv2.waitKey(0)
        # 图片宽度、高度、通道数
        width, height, channel = image.shape
        # 网络输入
        net_input = np.reshape(image/255, (1, width, height, channel))
        # 亮度通道变换
        new_image = self.sess.run(self.output, feed_dict={self.data:net_input})
        # reshape
        new_image = new_image * 255
        new_image = np.reshape(new_image, (width, height, channel))
        # cv2.imshow('1', new_image)
        # cv2.waitKey(0)
        # 转化回模式
        # new_image = cv2.cvtColor(new_image, cv2.COLOR_YCrCb2BGR)
        # 返回图片
        return new_image


    def conv2d(self,input,filter):
        return tf.nn.conv2d(input=input,filter=filter,strides=[1,1,1,1],padding="SAME")


    def read_data(self, path):
        with h5py.File(path, 'r') as file:
            try:
                data = file.get('train_data').value
                return data
            except:
                print('some wrong!')


    def process_origin_data(self, data, scale):
        train_labels = data
        train_data = []
        num, width, height, channel = data.shape
        data = np.split(data, num, axis=0)
        for i in range(len(data)):
            image = np.reshape(data[i],(width,height,channel))
            new_image = cv2.resize(image,(int(width*scale), int(height*scale)),interpolation=cv2.INTER_AREA)
            new_image = cv2.resize(new_image, (width, height), interpolation=cv2.INTER_CUBIC)
            train_data.append(new_image)

        train_data = np.array(train_data)

        return train_data, train_labels


    def get_batch(self, data, labels, batch_num):
        total = len(data)
        # 转化成数组

        items = np.array(np.arange(0, total))
        np.random.shuffle(items)

        batch_data, batch_labels, one_batch_data, one_batch_labels = [], [], [], []
        for i in range(total):
            n = items[i]
            one_batch_data.append(data[n]/255)
            one_batch_labels.append(labels[n]/255)

            if((i+1) % batch_num == 0):
                batch_data.append(np.array(one_batch_data))
                batch_labels.append(np.array(one_batch_labels))
                one_batch_data, one_batch_labels = [], []

            if(i+1 == total):
                batch_data.append(np.array(one_batch_data))
                batch_labels.append(np.array(one_batch_labels))
                one_batch_data, one_batch_labels = [], []

        return np.array(batch_data), np.array(batch_labels)

if __name__ == '__main__':
    with tf.Session() as sess:
        model = SRCNN(sess, './dataset.h5', epoch, batch_size, 1/3)
        # model.train()
        model.load_model()
        # new_image = model.predict('./images/interpolation/1.jpg')
        # cv2.imwrite('./images/output/1.jpg', new_image)
        # new_image = model.predict('./images/interpolation/2.png')
        # cv2.imwrite('./images/output/2.png', new_image)
        # new_image = model.predict('./images/interpolation/5.jpg')
        # cv2.imwrite('./images/output/5.jpg', new_image)

        new_image = model.predict('./images/interpolation/63.jpg')
        cv2.imwrite('./images/output/63.jpg', new_image)

