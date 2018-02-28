# -*- coding: utf-8 -*-

import IDataLoader

import os
import cv2
import numpy
import datetime
import random

from keras.utils import np_utils

from KEF.Models import Data
from random import shuffle

from imgaug import augmenters as iaa
import imgaug as ia

import librosa

import csv

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


class CrossmodalLoader_A5Experiments_LipsArms_Framebased_SoundSpectogram(IDataLoader.IDataLoader):
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


    def slice_signal(self, signal, sliceSize, stride=0.5):
        """ Return windows of the given signal by sweeping in stride fractions
            of window
        """
        #print "Dimension:", signal.shape

        #print "Audio:", signal
        #print "Audio:", signal.shape


        #sliceSize = (2 ** 14) * sliceSize
        sliceSize = 16000 * sliceSize
        #sliceSize = 44100 * sliceSize
        slices = []
        currentFrame = 0

        # print "------"
        # print "Total Signal Size:", len(signal)
        # print "In Seconds: ", len(signal) / sliceSize
        # print "Each Slide: ", sliceSize
        # print "Number of Slices: ", len(signal) / sliceSize*stride
        # print "------"



        while currentFrame+sliceSize < len(signal):
            currentSlide = signal[currentFrame:int(currentFrame+sliceSize)]
            slices.append(currentSlide)
            currentFrame = int(currentFrame+sliceSize*stride)
            #print "Shape Current slide:", len(currentSlide)

        #print "Shape Slices:", len(slices)
        #raw_input("here")
        return numpy.array(slices)


        assert signal.ndim == 1, signal.ndim
        n_samples = signal.shape[0]
        offset = int(window_size * stride)
        slices = []
        for beg_i, end_i in zip(range(0, n_samples, offset),
                                range(window_size, n_samples + offset,
                                      offset)):
            if end_i - beg_i < window_size:
                break
            slice_ = signal[beg_i:end_i]
            if slice_.shape[0] == window_size:
                slices.append(slice_)
        return numpy.array(slices, dtype=numpy.int32)




    def preEmphasis(self, signal, coeff=0.95):


        x = numpy.array(signal)
        x0 = numpy.reshape(x[0], [1, ])
        diff = x[1:] - coeff * x[:-1]
        concat = numpy.concatenate([x0, diff], 0)
        return concat


    def preProcessAudio(self, dataLocation, augment=False):

        assert (
        not dataLocation == None or not self.preProcessingProperties == ""), "No data or parameter sent to preprocess!"

        assert (len(self.preProcessingProperties[
                        0]) == 2), "The preprocess have the wrong shape. Right shape: ([imgSize,imgSize])."



        fftsize = 1024
        hop_length = 512

        wav_data, sr = librosa.load(dataLocation, mono=False, sr=16000)

        wav_data1 = numpy.array(wav_data[0])
        wav_data2 = numpy.array(wav_data[1])

        signals1 = [wav_data1]
        signals2 = [wav_data2]


        signals = []
        for wav_data_index in range(len(signals1)):

            D = librosa.stft(signals1[wav_data_index], fftsize, hop_length=hop_length)
            magD1 = numpy.abs(D)

            D = librosa.stft(signals2[wav_data_index], fftsize, hop_length=hop_length)
            magD2 = numpy.abs(D)


            magD1 = numpy.array(cv2.resize(magD1, (26, 512)))
            #magD1 = numpy.array(cv2.resize(magD1, (42, 512)))
            magD1 = numpy.expand_dims(magD1, axis=0)

            magD2 = numpy.array(cv2.resize(magD2, (26, 512)))
            #magD1 = numpy.array(cv2.resize(magD1, (42, 512)))
            magD2 = numpy.expand_dims(magD2, axis=0)

            magD = numpy.concatenate((magD1,magD2),axis=0)



            signals.append(magD)


        return numpy.array(signals)


    def preProcess(self, dataLocation, imageSize = (128,128), grayScale = False, augment=False):

        assert (
            not dataLocation == None or not self.preProcessingProperties == ""), "No data or parameter sent to preprocess!"

        assert (len(self.preProcessingProperties[
                        0]) == 2), "The preprocess have the wrong shape. Right shape: ([imgSize,imgSize])."

        # videoClip = VideoFileClip(dataLocation)


        #imageSize = self.preProcessingProperties[0]

        #grayScale = self.preProcessingProperties[1]

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



        return dataList

    def loadData(self, dataLipsFolder, dataArmsFolder, dataSoundFolder, dataParticipantsFolder,participantNumbers, augmentData):

        def shuffle_unison(a, b):
            assert len(a) == len(b)
            p = numpy.random.permutation(len(a))
            return a[p], b[p]

        assert (not dataParticipantsFolder == None or not dataParticipantsFolder == ""), "Empty Participants Folder!"

        dataX = []
        dataLabels = []



        readingFrom = ""
        participantsCSVs = os.listdir(dataParticipantsFolder)
        participantNumber = 0
        for participantCSV in participantsCSVs:
            loadedDataPoints = 0
            time = datetime.datetime.now()
            if participantNumber in participantNumbers:

                with open(dataParticipantsFolder+"/"+ participantCSV, 'rb') as csvfile:
                    readingFrom = dataParticipantsFolder+"/"+ participantCSV
                    #print "Reading From:", readingFrom
                    #raw_input(("here"))
                    reader = csv.reader(csvfile)
                    rownum = 0
                    for row in reader:
                        #print "Row:", row
                        if rownum >= 1:

                          #print "Row:", row
                          audioLabel = row[6]
                          lipsLabel = row[7]
                          armsLabel = row[8]
                          congruencyLabel = row[9]
                          conditionLabel = row[10]
                          phonemeLabel = row[-2]
                          decisionLabel = row[-1]

                          if not decisionLabel == "Na":
                                # partialLabel = str(audioLabel) + "_" + str(lipsLabel) + "_" + str(
                                #   armsLabel) + "_" + str(congruencyLabel) + "_" + str(conditionLabel) + "_" + str(
                                #   phonemeLabel)

                                partialLabel = str(audioLabel) + "_" + str(lipsLabel) + "_" + str(
                                    armsLabel)


                                #Load Arms
                                armsFolderList = os.listdir(dataArmsFolder +"/"+str(armsLabel)+"/")

                                matchingarms = [s for s in armsFolderList if partialLabel in s]
                                shuffle(matchingarms)

                                armDataPoint = self.preProcess(dataArmsFolder +"/"+str(armsLabel)+"/"+matchingarms[0],imageSize=(80,60), grayScale=True, augment=False)


                                #Load Lips
                                lipsFolderList = os.listdir(dataLipsFolder+"/"+str(lipsLabel)+"/")
                                matchingLips = [s for s in lipsFolderList if partialLabel in s]
                                shuffle(matchingLips)

                                lipsImagesList = self.orderDataFolder(dataLipsFolder+"/"+str(lipsLabel)+"/"+matchingLips[0])
                                lipsDataPoint = []
                                for lipsImage in lipsImagesList:
                                    #print "Opening:", dataLipsFolder + "/" + str(lipsLabel) + "/" + matchingLips[0] + "/" + lipsImage
                                    lipDataPoint = self.preProcess(
                                        dataLipsFolder + "/" + str(lipsLabel) + "/" + matchingLips[0] + "/" + lipsImage,
                                        imageSize=(120, 120), grayScale=True, augment=False)
                                    lipsDataPoint.append(lipDataPoint)

                                lipsDataPoint = numpy.array(lipsDataPoint)
                                lipsDataPoint = lipsDataPoint.squeeze(axis=1)

                                #Load Sound

                                soundFolderList = os.listdir(dataSoundFolder + "/" + str(audioLabel) + "/")
                                matchingAudio = [s for s in soundFolderList if partialLabel in s]
                                shuffle(matchingAudio)

                                # print "Sounds:", matchingAudio
                                # print "List:", dataSoundFolder + "/" + str(audioLabel) + "/"
                                # print "Label:", partialLabel
                                # print "Opening:", dataSoundFolder + "/" + str(audioLabel) + "/"+matchingAudio[0]

                                audioDataPoint = self.preProcessAudio(dataSoundFolder + "/" + str(audioLabel) + "/"+matchingAudio[0])[0]


                                # print "Shape Arms:", armDataPoint.shape
                                # print "Shape Lips:", lipsDataPoint.shape
                                # print "Shape Audio:", audioDataPoint.shape
                                # print "Label:", decisionLabel


                                dataX.append([numpy.array(armDataPoint).astype('float32'), numpy.array(lipsDataPoint).astype('float32'),numpy.array(audioDataPoint).astype('float32')])
                                dataLabels.append(int(decisionLabel)-1)
                                loadedDataPoints = loadedDataPoints+1

                        rownum += 1
                        # if rownum == 30:
                        #     break
                self.logManager.write(
                                        "--- Participant: " + str(participantNumber) + "(" + str(
                                            loadedDataPoints) + " Data points - " + str(
                                            (datetime.datetime.now() - time).total_seconds()) + " seconds" + ")")


            participantNumber = participantNumber + 1



        dataLabels = np_utils.to_categorical(dataLabels, 4)


        dataX = numpy.array(dataX)
        #print "dataX shape:", numpy.array(dataX).shape
        # dataX = numpy.swapaxes(dataX, 1, 2)
        #dataX = numpy.squeeze(dataX, axis=2)

        # dataLabels = numpy.array(dataLabels).reshape((len(dataLabels), 2))

 #       print "Shape DataX:", dataX.shape
#        dataX = dataX.astype('float32')

        #dataX, dataLabels = shuffle_unison(dataX, dataLabels)

        print "dataX shape:", numpy.array(dataX).shape
        print "dataY shape:", numpy.array(dataLabels).shape
        # raw_input("here")

        dataX = numpy.array(dataX)
        dataLabels = numpy.array(dataLabels)

        return Data.Data(dataX, dataLabels, []), readingFrom

    def loadTrainData(self, dataLipsFolder, dataArmsFolder, dataSoundFolder, dataParticipantsFolder, participantNumbers, augmentData=False):
        #print "participant:", participantNumbers
        self.logManager.newLogSession("Training Data")
        self.logManager.write("Loading Lips From: " + dataLipsFolder)
        self.logManager.write("Loading Arms From: " + dataArmsFolder)
        self.logManager.write("Loading Sounds From: " + dataSoundFolder)
        self.logManager.write("Loading Participants From: " + dataParticipantsFolder)
        self.logManager.write("Loading Participants: " + str(participantNumbers))

        self.logManager.write("Preprocessing Properties: " + str(self.preProcessingProperties))
        self._dataTrain = self.loadData(dataLipsFolder, dataArmsFolder, dataSoundFolder, dataParticipantsFolder, participantNumbers, augmentData)[0]

        self.logManager.write("Total data points: " + str(len(self.dataTrain.dataX)))
        self.logManager.write("Data points shape: " + str(numpy.array(self.dataTrain.dataX).shape))
        self.logManager.write("Data labels shape: " + str(numpy.array(self.dataTrain.dataY).shape))
        self.logManager.endLogSession()

    def loadTestData(self, dataLipsFolder, dataArmsFolder, dataSoundFolder, dataParticipantsFolder, participantNumbers, augmentData=False):
        self.logManager.newLogSession("Training Data")
        self.logManager.write("Loading Lips From: " + dataLipsFolder)
        self.logManager.write("Loading Arms From: " + dataArmsFolder)
        self.logManager.write("Loading Sounds From: " + dataSoundFolder)
        self.logManager.write("Loading Participants From: " + dataParticipantsFolder)
        self.logManager.write("Loading Participants: " + str(participantNumbers))

        self.logManager.write("Preprocessing Properties: " + str(self.preProcessingProperties))
        self._dataTest, participantCSV = self.loadData(dataLipsFolder, dataArmsFolder, dataSoundFolder, dataParticipantsFolder, participantNumbers, augmentData)
        self.logManager.write("Total data points: " + str(len(self.dataTest.dataX)))
        self.logManager.write("Data points shape: " + str(numpy.array(self.dataTest.dataX).shape))
        self.logManager.write("Data labels shape: " + str(numpy.array(self.dataTest.dataY).shape))
        self.logManager.endLogSession()
        return participantCSV

    def loadValidationData(self, dataLipsFolder, dataArmsFolder, dataSoundFolder, dataParticipantsFolder, participantNumbers, augmentData=False):
        self.logManager.newLogSession("Training Data")
        self.logManager.write("Loading Lips From: " + dataLipsFolder)
        self.logManager.write("Loading Arms From: " + dataArmsFolder)
        self.logManager.write("Loading Sounds From: " + dataSoundFolder)
        self.logManager.write("Loading Participants From: " + dataParticipantsFolder)
        self.logManager.write("Loading Participants: " + str(participantNumbers))

        self.logManager.write("Preprocessing Properties: " + str(self.preProcessingProperties))
        self._dataTest, participantCSV = self.loadData(dataLipsFolder, dataArmsFolder, dataSoundFolder, dataParticipantsFolder, participantNumbers, augmentData)
        self.logManager.write("Total data points: " + str(len(self.dataValidation.dataX)))
        self.logManager.write("Data points shape: " + str(numpy.array(self.dataValidation.dataX).shape))
        self.logManager.write("Data labels shape: " + str(numpy.array(self.dataValidation.dataY).shape))
        self.logManager.endLogSession()
        return participantCSV

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