import tensorflow as tf
import numpy as np

def read_data():
    # read data
    raw_data = []
    with open ('./data.txt','r') as f:
        line = 1
        while(line):
            line = f.readline()
            a_data = line.split('\t')
            a_data = data_process(a_data)
            raw_data.append(a_data)
    return raw_data[0:105], raw_data[105:149]


    # process data
    # train_data = raw_data[0:105]
    # test_data = raw_data[105:149]

def data_process(a_data):
    try:
        if a_data[5] == 'A\n':
            a_data[5] = [1,0,0]
        elif a_data[5] == 'B\n':
            a_data[5] = [0,1,0]
        elif a_data[5] == 'C\n':
            a_data[5] = [0,0,1]
        return a_data
    except:
        return a_data

def add_layer(inputs, input_size, output_size, activation_function=None, dropout_rate=1):
    weights = tf.Variable(tf.random_normal([input_size,output_size]))
    bias = tf.Variable(tf.zeros([1,output_size]) + 0.1)
    result = tf.matmul(inputs, weights) + bias
    result = tf.nn.dropout(result,keep_prob=dropout_rate)
    if activation_function == None:
        outputs = result
    else:
        outputs = activation_function(result)
    return outputs

xs = tf.placeholder(tf.float32,[None,4]) # 五个特征
ys = tf.placeholder(tf.float32,[None,3]) # 三个类别

hideen_layer = add_layer(xs,4,10,activation_function=tf.nn.relu)
prediction = add_layer(hideen_layer,10,3,activation_function=tf.nn.softmax)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    train_data, other_data = read_data()

    train_x = []
    train_y = []
    for a_data in train_data:
        train_x.append(np.array(a_data[1:5]))
        train_y.append(np.array(a_data[5]))
    # train 1000 times
    for i in range(1000):
        sess.run(train_step,feed_dict={xs:train_x,ys:train_y})
        print(sess.run(cross_entropy, feed_dict={xs:train_x,ys:train_y}))

    # predict
    other_x = []
    for a_data in other_data:
        other_x.append(np.array(a_data[1:5]))

    result = sess.run(prediction,feed_dict={xs:other_x})
    final_result = sess.run(tf.argmax(result,1))

    # write data
    for i in range(len(other_data)):
        num = final_result[i]
        if(num == 0):
            result = "A"
            other_data[i].append('\t'+ result + '\n')
        elif(num == 1):
            result = "B"
            other_data[i].append('\t' + result + '\n')
        elif (num == 2):
            result = "C"
            other_data[i].append('\t' + result + '\n')

    print(other_data)

    with open('./result.txt','a') as f:
    	f.write('')
    # output
    with open('./result.txt','a') as f:
        for a_data in other_data:
            for a in a_data:
                f.write(a.strip() + '\t')
            f.write('\n')








