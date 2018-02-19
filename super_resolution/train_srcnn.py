import tensorflow as tf
import h5py


learning_rate = 1e-3

class SRCNN(object):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.input = tf.placeholder(tf.float32, [None, None, None, 3], name='images')
        self.build_graph()


    def build_graph(self):
        weights = {'w1': tf.Variable(tf.random_normal([9, 9, 3, 64], stddev=1e-3)),
                   'w2': tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3)),
                   'w3': tf.Variable(tf.random_normal([5, 5, 32, 1], stddev=1e-3))}

        biases = {'b1': tf.Variable(tf.zeros([64])),
                  'b2': tf.Variable(tf.zeros([32])),
                  'b3': tf.Variable(tf.zeros([1]))}

        conv1 = tf.nn.relu(self.conv2d(self.input, weights['w1']) + biases['b1'])
        conv2 = tf.nn.relu(self.conv2d(conv1, weights['w2']) + biases['b2'])
        conv3 = self.conv2d(conv2, weights['w3']) + biases['b3']

        self.output = conv3



    def update(self, iterNum):
        train_data, train_labels, test_data, test_labels = self.read_data(self.dataset_path)
        self.loss = tf.reduce_mean(tf.square(test_labels - self.output))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        with tf.Session() as sess:
            for i in range(iterNum):
                print('iter:' + str(i))
                sess.run(tf.global_variables_initializer())
                sess.run(self.optimizer, feed_dict={self.input:test_data})
                print(sess.run(self.loss, feed_dict={self.input:test_data}))



    def conv2d(self,input,filter):
        return tf.nn.conv2d(input=input,filter=filter,strides=[1,1,1,1],padding="SAME")


    def read_data(self, path):
        with h5py.File(path, 'r') as file:
            try:
                train_data = file.get('train_data')
                train_labels = file.get('train_labels')
                test_data = file.get('test_data')
                test_labels = file.get('test_labels')
                return train_data.value, train_labels.value, test_data.value, test_labels.value
            except:
                print('some wrong!')



if __name__ == '__main__':
    model = SRCNN('./dataset.h5')
    model.update(10)
