import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
import os
import cv2
import numpy as np
import time

DATA_PATH = './images/train/down_sampling'
LABELS_PATH = './images/train/cut'
MODEL_PATH = './model/params.pkl'
FACTOR = 4
TRAIN_RATIO = 9/10
BATCH_SIZE = 300
LR = 1e-4

class ESPCN_Model(torch.nn.Module):
    def __init__(self, factor):
        super(ESPCN_Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 32, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(32, factor*factor*3, 3,padding=1)
        self.pixel_shuffle = torch.nn.PixelShuffle(factor)



    def forward(self, input):
        x = F.tanh(self.conv1(input))
        x = F.tanh(self.conv2(x))
        x = self.pixel_shuffle(self.conv3(x))
        return x

    def initialize_weights(self):
        pass



class ESPCN(object):
    def __init__(self, data_path, labels_path, model_path, factor, train_ratio):
        self.data_path = data_path
        self.labels_path = labels_path
        self.model_path = model_path
        self.train_ratio = train_ratio
        self.train_loader = []
        self.test = []
        # build net
        self.model = ESPCN_Model(factor)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.loss_func = torch.nn.MSELoss()




    def train(self, iter_num):
        self.load_data(self.data_path, self.labels_path)
        for epoch in range(iter_num):
            print('--------------------------------------')
            print('epoch:', epoch)
            print('--------------------------------------')
            start_time = time.time()

            for batch, (x, y) in enumerate(self.train_loader):
                print('training batch:',batch)

                b_x = Variable(x.type(torch.FloatTensor))
                b_y = Variable(y.type(torch.FloatTensor))


                # forward
                output = self.model(b_x)
                # compute loss
                train_loss = self.loss_func(output, b_y)

                print('train_loss:', train_loss.data.numpy()[0])


                # back propagation
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()

            end_time = time.time()
            cost_time = end_time - start_time

            predict = self.model(Variable(self.test[0].type(torch.FloatTensor)))
            predict_loss = self.loss_func(predict, Variable(self.test[1].type(torch.FloatTensor)))

            print('\ntest dataset loss:', predict_loss.data.numpy()[0])
            print('cost_time:', cost_time)

            if(epoch%5 == 0):
                torch.save(self.model.state_dict(), self.model_path)




    def predict(self, from_path, dest_path):
        self.model.load_state_dict(torch.load(self.model_path))
        if os.path.exists(from_path):
            file_list = os.listdir(from_path)
            for file in file_list:
                img_path = os.path.join(from_path, file)
                image = cv2.imread(img_path) / 255
                image = np.moveaxis(image,2,0)
                image = np.reshape(image,(1,) + image.shape)

                input = torch.from_numpy(image).type(torch.FloatTensor)
                predict = self.model(Variable(input))

                predict = predict.data.numpy()
                n, channel, width, height = predict.shape
                predict = np.reshape(predict, [channel, width, height])
                predict = np.moveaxis(predict,0,-1) * 255
                cv2.imwrite(os.path.join(dest_path, file), predict)







    def load_data(self, data_path, labels_path):
        data = []
        labels = []

        if os.path.exists(data_path):
            file_list = os.listdir(data_path)
            for file in file_list:
                a_data_path = os.path.join(data_path, file)
                a_label_path = os.path.join(labels_path, file)

                data_image = cv2.imread(a_data_path) / 255
                label_image = cv2.imread(a_label_path) / 255

                # move axis to [channel, width, height]
                data_image = np.moveaxis(data_image, 2, 0)
                label_image = np.moveaxis(label_image, 2, 0)


                data.append(data_image)
                labels.append(label_image)


            total = len(labels)
            split = int(total * self.train_ratio)
            train_data = data[0:split]
            train_labels = labels[0:split]
            test_data = data[split + 1:]
            test_labels = labels[split + 1:]

            train_data, train_labels = torch.from_numpy(np.array(train_data)), torch.from_numpy(np.array(train_labels))
            # train data
            train_dataset = Data.TensorDataset(data_tensor=train_data, target_tensor=train_labels)
            self.train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
            # test data
            self.test.append(torch.from_numpy(np.array(test_data)))
            self.test.append(torch.from_numpy(np.array(test_labels)))
        else:
            print('file do not exist!')
            return None






if __name__ == '__main__':
    model = ESPCN(DATA_PATH, LABELS_PATH, MODEL_PATH, FACTOR, TRAIN_RATIO)
    model.train(100)
    model.predict('./images/test/down_sampling', './images/test/predict')





