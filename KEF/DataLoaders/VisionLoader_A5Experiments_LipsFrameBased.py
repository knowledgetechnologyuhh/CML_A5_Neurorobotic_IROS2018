# -*- coding: utf-8 -*-

import IDataLoader

import os
import cv2
import numpy
import datetime
import random

from keras.utils import np_utils

from KEF.Models import Data

from imgaug import augmenters as iaa
import imgaug as ia

st = lambda aug: iaa.Sometimes(0.5, aug)

seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    # st(iaa.Add((-10, 10), per_channel=0.5)),
    # st(iaa.Multiply((0.5, 1.5), per_channel=0.5)),
    # st(iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)),
    st(iaa.Affine(
        scale={"x": (0.9, 1.10), "y": (0.9, 1.10)},
        # scale images to 80-120% of their size, individually per axis
        translate_px={"x": (-5, 5), "y": (-5, 5)},  # translate by -16 to +16 pixels (per axis)
        rotate=(-10, 10),  # rotate by -45 to +45 degrees
        shear=(-3, 3),  # shear by -16 to +16 degrees
        order=3,  # use any of scikit-image's interpolation methods
        cval=(0.0, 1.0),  # if mode is constant, use a cval between 0 and 1.0
        mode="constant"
    )),
], random_order=True)


class VisionLoader_A5Experiments_LipsFrameBased(IDataLoader.IDataLoader):
    numberOfAugmentedSamples = 10
    framesPerSecond = 25
    stride = 25

    @property
    def logManager(self):
        return self._logManager

    @property
    def dataTrain(self):
        return self._dataTrain

    @property
    def dataValidation(self):
        return self._dataValidation

    @property
    def dataTest(self):
        return self._dataTest

    @property
    def preProcessingProperties(self):
        return self._preProcessingProperties

    def __init__(self, logManager, preProcessingProperties=None, fps=25, stride=25):

        assert (not logManager == None), "No Log Manager was sent!"

        self._preProcessingProperties = preProcessingProperties

        self._dataTrain = None
        self._dataTest = None
        self._dataValidation = None
        self._logManager = logManager
        self.framesPerSecond = fps
        self.stride = stride

    def dataAugmentation(self, dataPoint):

        samples = []
        samples.append(dataPoint)
        for i in range(self.numberOfAugmentedSamples):
            samples.append(seq.augment_image(dataPoint))

        return numpy.array(samples)

    def preProcess(self, dataLocation, augment=False):

        assert (
            not dataLocation == None or not self.preProcessingProperties == ""), "No data or parameter sent to preprocess!"

        assert (len(self.preProcessingProperties[
                        0]) == 2), "The preprocess have the wrong shape. Right shape: ([imgSize,imgSize])."

        # videoClip = VideoFileClip(dataLocation)


        imageSize = self.preProcessingProperties[0]

        grayScale = self.preProcessingProperties[1]

        frame = cv2.imread(dataLocation)

        data = numpy.array(cv2.resize(frame, imageSize))

        if grayScale:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

        if augment:
            data = self.dataAugmentation(data)

            # print "Shape:", numpy.array(data).shape

        if grayScale:

            if augment:
                data = numpy.expand_dims(data, axis=1)
            else:
                data = numpy.expand_dims(data, axis=0)


        else:

            if augment:
                data = numpy.swapaxes(data, 2, 3)
                data = numpy.swapaxes(data, 1, 2)
            else:
                data = numpy.swapaxes(data, 1, 2)
                data = numpy.swapaxes(data, 0, 1)

        data = data.astype('float32')

        data /= 255

        data = numpy.array(data)

        # print "Total Frames:", len(dataSequence)


        return data

    def orderClassesFolder(self, folder):

        classes = os.listdir(folder)

        return classes

    def orderDataFolder(self, folder):

        dataList = os.listdir(folder)

        dataList = sorted(dataList, key=lambda x: int(x.split(".")[0]))

        # for item in dataList:
        #    print item.split("__")

        # print "size:", len(dataList)
        # print "Folder:", folder
        # print "DataList:", dataList
        # raw_input("here")
        # print dataList


        return dataList

    def loadData(self, dataFolder, augment):

        def shuffle_unison(a, b):
            assert len(a) == len(b)
            p = numpy.random.permutation(len(a))
            return a[p], b[p]

        assert (not dataFolder == None or not dataFolder == ""), "Empty Data Folder!"

        dataX = []
        dataLabels = []
        classesDictionary = []

        classes = self.orderClassesFolder(dataFolder + "/")
        self.logManager.write("Videos reading order: " + str(classes))

        lastImage = None

        numberOfVideos = 0
        classNumber = 0
        for c in classes:
            videos = self.orderClassesFolder(dataFolder + "/" + str(c) + "/")
            loadedDataPoints = 0
            time = datetime.datetime.now()

            for v in videos:

                # if numberOfVideos < 5:
                numberOfVideos = numberOfVideos + 1

                # print dataFolder + "/" + v+"/"+dataPointLocation
                dataFrames = self.orderDataFolder(dataFolder + "/" + str(c) + "/" + v)
                sequenceFrames = []


                for i in range(len(dataFrames)):
                    dataFrame = dataFrames[i]

                    try:
                        dataPoint = self.preProcess(dataFolder + "/" + str(c) + "/" + v + "/" + dataFrame,
                                                    augment)

                        if augment:
                            lastImage = dataPoint[0]
                        else:
                            lastImage = dataPoint
                    except:
                        print "Error:", dataFolder + "/" + str(c) + "/" + v + "/" + dataFrame

                        if augment:
                            dataPoint = [lastImage]
                        else:
                            dataPoint = lastImage
                    #print "DataPoint Shape:", dataPoint.shape
                    sequenceFrames.append(dataPoint)

                sequenceFrames = numpy.array(sequenceFrames)
                # print "Shape:", sequenceFrames.shape
                dataX.append(sequenceFrames)
                dataLabels.append(classNumber)
                loadedDataPoints = loadedDataPoints + 1

            classNumber = classNumber + 1

            self.logManager.write(
                "--- Arm Position: " + str(c) + "(" + str(loadedDataPoints) + " Data points - " + str(
                    (datetime.datetime.now() - time).total_seconds()) + " seconds" + ")")

        dataLabels = np_utils.to_categorical(dataLabels, classNumber)


        dataX = numpy.array(dataX)
        print "dataX shape:", numpy.array(dataX).shape
        # dataX = numpy.swapaxes(dataX, 1, 2)
        dataX = numpy.squeeze(dataX, axis=2)

        # dataLabels = numpy.array(dataLabels).reshape((len(dataLabels), 2))


        dataX = dataX.astype('float32')

        dataX, dataLabels = shuffle_unison(dataX, dataLabels)

        print "dataX shape:", numpy.array(dataX).shape
        print "dataY shape:", numpy.array(dataLabels).shape
        # raw_input("here")

        dataX = numpy.array(dataX)
        dataLabels = numpy.array(dataLabels)

        return Data.Data(dataX, dataLabels, classesDictionary)

    def loadTrainData(self, dataFolder, augmentData=False):
        self.logManager.newLogSession("Training Data")
        self.logManager.write("Loading From: " + dataFolder)
        self.logManager.write("Preprocessing Properties: " + str(self.preProcessingProperties))
        self._dataTrain = self.loadData(dataFolder, augmentData)

        self.logManager.write("Total data points: " + str(len(self.dataTrain.dataX)))
        self.logManager.write("Data points shape: " + str(numpy.array(self.dataTrain.dataX).shape))
        self.logManager.write("Data labels shape: " + str(numpy.array(self.dataTrain.dataY).shape))
        self.logManager.endLogSession()

    def loadTestData(self, dataFolder, augmentData=False):
        self.logManager.newLogSession("Testing Data")
        self.logManager.write("Loading From: " + dataFolder)
        self.logManager.write("Preprocessing Properties: " + str(self.preProcessingProperties))
        self._dataTest = self.loadData(dataFolder, augmentData)
        self.logManager.write("Total data points: " + str(len(self.dataTest.dataX)))
        self.logManager.write("Data points shape: " + str(numpy.array(self.dataTest.dataX).shape))
        self.logManager.write("Data labels shape: " + str(numpy.array(self.dataTest.dataY).shape))
        self.logManager.endLogSession()

    def loadValidationData(self, dataFolder, augmentData=False):
        self.logManager.newLogSession("Validation Data")
        self.logManager.write("Loading From: " + dataFolder)
        self.logManager.write("Preprocessing Properties: " + str(self.preProcessingProperties))
        self._dataValidation = self.loadData(dataFolder, augmentData)
        self.logManager.write("Total data points: " + str(len(self.dataValidation.dataX)))
        self.logManager.write("Data points shape: " + str(numpy.array(self.dataValidation.dataX).shape))
        self.logManager.write("Data labels shape: " + str(numpy.array(self.dataValidation.dataY).shape))
        self.logManager.endLogSession()

    def saveData(self, folder):
        pass

    def shuffleData(self, dataX, dataY):

        positions = []
        for p in range(len(dataX)):
            positions.append(p)

        random.shuffle(positions)

        newInputs = []
        newOutputs = []
        for p in positions:
            newInputs.append(dataX[p])
            newOutputs.append(dataY[p])

        return (newInputs, newOutputs)

    def loadTrainTestValidationData(self, folder, percentage):
        pass

    def loadNFoldValidationData(self, folder, NFold):
        pass