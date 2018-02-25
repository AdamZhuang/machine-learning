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

    def saveToH5File(self, dataFolder, patchNum, savePath):
        data = []

        fileNames = utils.eachFile(dataFolder)
        for fileName in fileNames:
            data_image = cv2.imread(fileName)
            # every image sample patchNum patch
            width, height, channel = data_image.shape
            for i in range(patchNum):
                m = int(np.random.rand() * (width-80))
                n = int(np.random.rand() * (height-80))
                patch = data_image[m:m+80, n:n+80]
                data.append(patch)

        data = np.array(data)
        print('total patch:', len(data))

        with h5py.File(savePath, 'w') as file:
            file.create_dataset('train_data', data=data)
            print('DataSet saved.')



    def cut_images(self, srcFolder, destFolder):
        num = 0
        fileNames = utils.eachFile(srcFolder)
        for fileName in fileNames:
            image = cv2.imread(fileName)

            if(image is None):
                utils.deleteFile(fileName)
                continue

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
    dataProcessor.reduce_size('./images/cut', './images/lowResolution', 1/5)
    dataProcessor.restore_size('./images/lowResolution', './images/interpolation','./images/cut')
    # dataProcessor.saveToH5File('./images/cut', 50, './dataset.h5')
