import cv2
import utils
import os
import re
import h5py
import numpy as np

class DataProcessor(object):
    def __init__(self):
        pass

    def reduce_size(self, srcFolder, destFolder, magnification):
        if (magnification >= 1):
            print('Magnification must be less than 1！')
            return False

        fileNames = utils.eachFile(srcFolder)
        for fileName in fileNames:
            image = cv2.imread(fileName)

            # 删除不可用图片
            if(image is None):
                utils.deleteFile(fileName)
                continue

            width, height, channel = image.shape
            # 向下取整缩小
            output = cv2.resize(image, (int(width * magnification), int(height * magnification)),
                                interpolation=cv2.INTER_AREA)

            fileName = re.findall(srcFolder+'/(.*)' ,fileName)[0]
            newFilePath = os.path.join(destFolder, fileName)
            cv2.imwrite(newFilePath, output)


    def restore_size(self, srcFolder, destFolder, referFolder):
        fileNames = utils.eachFile(srcFolder)
        for fileName in fileNames:
            referFileName = re.findall(srcFolder + '/(.*)', fileName)[0]
            referFilePath = os.path.join(referFolder, referFileName)
            referImage = cv2.imread(referFilePath)

            if(referImage is None):
                print(referFilePath)
                # continue

            width, height, channel = referImage.shape
            # reshape to original size
            src_image = cv2.imread(fileName)
            # height and width should reverse, and I don't not why
            output = cv2.resize(src_image, (height, width),
                                interpolation=cv2.INTER_CUBIC)

            fileName = re.findall(srcFolder + '/(.*)', fileName)[0]
            newFilePath = os.path.join(destFolder, fileName)
            cv2.imwrite(newFilePath, output)

    def saveToH5File(self, dataFolder, labelsFolder, savePath):
        data = []
        labels = []

        fileNames = utils.eachFile(dataFolder)
        for fileName in fileNames:
            print(fileName)
            data_image = cv2.imread(fileName)
            data_image = cv2.cvtColor(data_image, cv2.COLOR_BGR2YCR_CB)
            data_image = data_image / 255
            data.append(data_image)


            referFileName = re.findall(dataFolder + '/(.*)', fileName)[0]
            referFilePath = os.path.join(labelsFolder, referFileName)

            print(referFilePath)
            referImage = cv2.imread(referFilePath)
            referImage = cv2.cvtColor(referImage, cv2.COLOR_BGR2YCR_CB)
            referImage = referImage / 255
            referImage = cv2.cvtColor(referImage, cv2.COLOR_YCrCb2BGR)
            print(referImage)
            labels.append(referImage)

        total = len(data)
        print(total)

        train_data = np.array(data[:int(total*4/5)])
        train_labels = np.array(labels[:int(total*4/5)])
        test_data = np.array(data[int(total*4/5):])
        test_labels = np.array(labels[int(total*4/5):])

        with h5py.File(savePath, 'w') as file:
            file.create_dataset('train_data', data=train_data)
            file.create_dataset('train_labels', data=train_labels)
            file.create_dataset('test_data', data=test_data)
            file.create_dataset('test_labels', data=test_labels)
            print('DataSet saved.')



    def cut_images(self, srcFolder, destFolder):
        num = 0
        fileNames = utils.eachFile(srcFolder)
        for fileName in fileNames:
            image = cv2.imread(fileName)

            width, height, channel = image.shape
            if(width > 400 and height > 400):
                num += 1
                print(num, width, height)
                image = cv2.resize(image, (400, 400))

                fileName = re.findall(srcFolder + '/(.*)', fileName)[0]
                newFilePath = os.path.join(destFolder, fileName)
                cv2.imwrite(newFilePath, image)





if __name__ == '__main__':
    # utils.deleteGif('./images/origin')

    dataProcessor = DataProcessor()
    # dataProcessor.cut_images('./images/origin', './images/cut')
    # dataProcessor.reduce_size('./images/cut', './images/lowResolution', 1/4)
    # dataProcessor.restore_size('./images/lowResolution', './images/interpolation','./images/cut')
    dataProcessor.saveToH5File('./images/interpolation', './images/cut', './dataset.h5')
