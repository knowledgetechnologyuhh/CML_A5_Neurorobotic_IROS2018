# -*- coding: utf-8 -*-

import IDataLoader

import os
import cv2
import numpy
import datetime
import random

import scipy.io.wavfile as wavfile
import scipy.signal
import resampy

from keras.utils import np_utils

from KEF.Models import Data

import librosa
import math

import pylab
import librosa.display

import python_speech_features.sigproc as audioProc

from scipy.io.wavfile import read, write


def audioFromSpectrogram(spectrogram, location, sr=16000, fftsize=1024, hop_length=512):
    spectrogram = numpy.squeeze(spectrogram, 2)
    spectrogram = numpy.array(cv2.resize(spectrogram, (94, 513)))

    #spectrogram *= spectrogram


    y_out = spsi(spectrogram, fftsize=fftsize, hop_length=hop_length)

    librosa.output.write_wav(location, y_out, sr, True)

    # p = numpy.angle(librosa.stft(y_out, fftsize, hop_length, center=False))
    # for i in range(50):
    #     S = magD * numpy.exp(1j * p)
    #     x = librosa.istft(S, hop_length,
    #                       center=True)  # Griffin Lim, assumes hann window; librosa only does one iteration?
    #     p = numpy.angle(librosa.stft(x, fftsize, hop_length, center=True))
    #
    # librosa.output.write_wav("//data/experimentsAudioSetBegan/" + str(audioNumber) + "test2.wav", y_out, sr)
    # librosa.output.write_wav("//data/experimentsAudioSetBegan/" + str(audioNumber) + "test2_norm.wav", y_out, sr, True)
    #
    # librosa.output.write_wav("//data/experimentsAudioSetBegan/" + str(audioNumber) + "original.wav", wav_data, sr, True)
    # audioNumber = audioNumber + 1



def saveSpectrogram(spectrogram, location):
    #print "Shape Before Squeeze:", spectrogram.shape#
    #spectrogram = numpy.squeeze(spectrogram, 2)
    #print "Shape After Squeeze:", spectrogram.shape

    #spectrogram = numpy.array(cv2.resize(spectrogram, (94, 513)))
    #spectrogram *= spectrogram
    #print "Shape After Squeeze:", spectrogram.shape



    #
    # spectrogram = numpy.squeeze(spectrogram, 0)
    #
    # print "Shape After Squeeze:", spectrogram.shape

    pylab.axis('off')  # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edge
    # S = librosa.feature.melspectrogram(y=magD, sr=sr)
    librosa.display.specshow(librosa.power_to_db(spectrogram, ref=numpy.max))
    pylab.savefig(location, bbox_inches=None, pad_inches=0)
    pylab.close()

def spsi(msgram, fftsize, hop_length):
    """
    Takes a 2D spectrogram ([freqs,frames]), the fft legnth (= widnow length) and the hope size (both in units of samples).
    Returns an audio signal.
    """

    numBins, numFrames = msgram.shape
    y_out = numpy.zeros(numFrames * hop_length + fftsize - hop_length)

    m_phase = numpy.zeros(numBins);
    m_win = scipy.signal.hanning(fftsize,
                                 sym=True)  # assumption here that hann was used to create the frames of the spectrogram

    # processes one frame of audio at a time
    for i in range(numFrames):
        m_mag = msgram[:, i]
        for j in range(1, numBins - 1):
            if (m_mag[j] > m_mag[j - 1] and m_mag[j] > m_mag[j + 1]):  # if j is a peak
                alpha = m_mag[j - 1];
                beta = m_mag[j];
                gamma = m_mag[j + 1];
                denom = alpha - 2 * beta + gamma;

                if (denom != 0):
                    p = 0.5 * (alpha - gamma) / denom;
                else:
                    p = 0;

                # phaseRate=2*math.pi*(j-1+p)/fftsize;    #adjusted phase rate
                phaseRate = 2 * math.pi * (j + p) / fftsize;  # adjusted phase rate
                m_phase[j] = m_phase[j] + hop_length * phaseRate;  # phase accumulator for this peak bin
                peakPhase = m_phase[j];

                # If actual peak is to the right of the bin freq
                if (p > 0):
                    # First bin to right has pi shift
                    bin = j + 1;
                    m_phase[bin] = peakPhase + math.pi;

                    # Bins to left have shift of pi
                    bin = j - 1;
                    while ((bin > 1) and (m_mag[bin] < m_mag[bin + 1])):  # until you reach the trough
                        m_phase[bin] = peakPhase + math.pi;
                        bin = bin - 1;

                    # Bins to the right (beyond the first) have 0 shift
                    bin = j + 2;
                    while ((bin < (numBins)) and (m_mag[bin] < m_mag[bin - 1])):
                        m_phase[bin] = peakPhase;
                        bin = bin + 1;

                # if actual peak is to the left of the bin frequency
                if (p < 0):
                    # First bin to left has pi shift
                    bin = j - 1;
                    m_phase[bin] = peakPhase + math.pi;

                    # and bins to the right of me - here I am stuck in the middle with you
                    bin = j + 1;
                    while ((bin < (numBins)) and (m_mag[bin] < m_mag[bin - 1])):
                        m_phase[bin] = peakPhase + math.pi;
                        bin = bin + 1;

                    # and further to the left have zero shift
                    bin = j - 2;
                    while ((bin > 1) and (m_mag[bin] < m_mag[bin + 1])):  # until trough
                        m_phase[bin] = peakPhase;
                        bin = bin - 1;

                        # end ops for peaks
        # end loop over fft bins with

        magphase = m_mag * numpy.exp(1j * m_phase)  # reconstruct with new phase (elementwise mult)
        magphase[0] = 0;
        magphase[numBins - 1] = 0  # remove dc and nyquist
        m_recon = numpy.concatenate([magphase, numpy.flip(numpy.conjugate(magphase[1:numBins - 1]), 0)])

        # overlap and add
        m_recon = numpy.real(numpy.fft.ifft(m_recon)) * m_win
        y_out[i * hop_length:i * hop_length + fftsize] += m_recon

    return y_out


def unProcess(dataSignal):
    #dataSignal = deEmphasis(dataSignal)

    #dataSignal *= 16000
    return dataSignal


def deEmphasis(signal, coeff=0.95):
    y = numpy.array(signal)
    if coeff <= 0:
        return y

    x = numpy.zeros(y.shape[0], dtype=numpy.float32)
    x[0] = y[0]
    for n in range(1, y.shape[0], 1):
        x[n] = coeff * x[n - 1] + y[n]
    return x


class AudioLoader_Spectrogram_A5Experiment(IDataLoader.IDataLoader):
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

    def __init__(self, logManager, preProcessingProperties=None, fps =25, stride=25):

        assert (not logManager == None), "No Log Manager was sent!"

        self._preProcessingProperties = preProcessingProperties

        self._dataTrain = None
        self._dataTest = None
        self._dataValidation = None
        self._logManager = logManager
        self.framesPerSecond = fps
        self.stride=stride


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


    def preProcess(self, dataLocation, augment=False):

        assert (
        not dataLocation == None or not self.preProcessingProperties == ""), "No data or parameter sent to preprocess!"

        assert (len(self.preProcessingProperties[
                        0]) == 2), "The preprocess have the wrong shape. Right shape: ([imgSize,imgSize])."


        #videoClip = VideoFileClip(dataLocation)
        #print "Audio:", dataLocation

        fftsize = 1024
        hop_length = 512

        wav_data, sr = librosa.load(dataLocation, mono=False, sr=16000)
        #print "Shape Wave_data:", wav_data.shape

        wav_data1 = numpy.array(wav_data[0])
        wav_data2 = numpy.array(wav_data[1])

        # D = librosa.stft(wav_data1, fftsize, hop_length=hop_length)
        # magD = numpy.abs(D)
        #
        # saveSpectrogram(magD,"/data/A5_Experiments/Spec1.png")
        #
        #
        # D = librosa.stft(wav_data2, fftsize, hop_length=hop_length)
        # magD = numpy.abs(D)
        # saveSpectrogram(magD, "/data/A5_Experiments/Spec2.png")

        #raw_input("here")


        #signals1 = self.slice_signal(wav_data1, 1, 1)
        #signals2 = self.slice_signal(wav_data2, 1, 1)

        # signals1 = [wav_data1[8000:-1]]
        # signals2 = [wav_data2[8000:-1]]

        signals1 = [wav_data1]
        signals2 = [wav_data2]


        # print "Shape Signal:", signals1.shape
        # print "Shape Signal2:", signals2.shape
        signals = []
        for wav_data_index in range(len(signals1)):
            #print "index:", signals1[wav_data_index]
            D = librosa.stft(signals1[wav_data_index], fftsize, hop_length=hop_length)
            magD1 = numpy.abs(D)

            D = librosa.stft(signals2[wav_data_index], fftsize, hop_length=hop_length)
            magD2 = numpy.abs(D)

            #magD1 = magD1 / magD1.max()
            #magD2 = magD2 / magD2.max()
            #magD = librosa.amplitude_to_db(magD, ref=numpy.max)
            #print "SHape:", magD1.shape
            #print "SHape:", magD2.shape
            #print "Max:", magD1.max()
            #print "Min:", magD1.min()




            magD1 = numpy.array(cv2.resize(magD1, (42, 512)))
            magD1 = numpy.expand_dims(magD1, axis=0)

            magD2 = numpy.array(cv2.resize(magD2, (42, 512)))
            magD2 = numpy.expand_dims(magD2, axis=0)

            magD = numpy.concatenate((magD1,magD2),axis=0)

            #magD /= 256
            #print "Shape:", magD.shape
            #raw_input("here")


            signals.append(magD)

        #raw_input("here")
        return numpy.array(signals)




    def orderClassesFolder(self, folder):

        classes = os.listdir(folder)

        return classes

    def orderDataFolder(self, folder):

        dataList = os.listdir(folder)


        dataList = sorted(dataList, key=lambda x: int(x.split(".")[0]))

        #for item in dataList:
        #    print item.split("__")

        #print "size:", len(dataList)
        #print "Folder:", folder
        #print "DataList:", dataList
        #raw_input("here")
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

        emotions = self.orderClassesFolder(dataFolder + "/")
        self.logManager.write("Emotions reading order: " + str(emotions))

        lastImage = None

        numberOfVideos = 0
        emotionNumber = 0
        for e in emotions:

            emotionNumber = emotionNumber+1
            loadedDataPoints = 0
            classesDictionary.append("'" + str(emotionNumber) + "':'" + str(e) + "',")


            time = datetime.datetime.now()

            #print dataFolder + "/" + v+"/"+dataPointLocation

            for audio in os.listdir(dataFolder + "/" + e+"/"):

             #if numberOfVideos < 30:
                dataPoint = self.preProcess(dataFolder + "/" + e+"/"+audio)

                for audio in dataPoint:
                    #print "SHape:", audio.shape
                    dataX.append(audio)
                    dataLabels.append(emotionNumber - 1)
                    loadedDataPoints = loadedDataPoints + 1
                numberOfVideos = numberOfVideos + 1


            self.logManager.write(
                "--- Emotion: " + str(e) + "(" + str(loadedDataPoints) + " Data points - " + str(
                    (datetime.datetime.now() - time).total_seconds()) + " seconds" + ")")


        #print "Labels before:", dataLabels
        dataLabels = np_utils.to_categorical(dataLabels, emotionNumber)
        #print "Labels After:", dataLabels
        dataX = numpy.array(dataX)
        #dataX = numpy.swapaxes(dataX, 1, 2)

        # print "Shape Labels:", dataLabels.shape
        # print "Shape DataX:", dataX.shape
        #dataLabels = numpy.array(dataLabels).reshape((len(dataLabels), 2))

#        dataX = dataX.astype('float32')


        dataX, dataLabels = shuffle_unison(dataX,dataLabels)

#        print "dataX shape:", numpy.array(dataX).shape
        #raw_input("here")

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
        #raw_input("here")

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


# -*- coding: utf-8 -*-

# import IDataLoader
#
# import os
# import cv2
# import numpy
# import datetime
# import random
# from moviepy.editor import *
#
# from keras.utils import np_utils
#
# from KEF.Models import Data
#
# from imgaug import augmenters as iaa
# import imgaug as ia
#
# st = lambda aug: iaa.Sometimes(0.5, aug)
#
# seq = iaa.Sequential([
#     iaa.Fliplr(0.5),
#     # st(iaa.Add((-10, 10), per_channel=0.5)),
#     # st(iaa.Multiply((0.5, 1.5), per_channel=0.5)),
#     # st(iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)),
#     st(iaa.Affine(
#         scale={"x": (0.9, 1.10), "y": (0.9, 1.10)},
#         # scale images to 80-120% of their size, individually per axis
#         translate_px={"x": (-5, 5), "y": (-5, 5)},  # translate by -16 to +16 pixels (per axis)
#         rotate=(-10, 10),  # rotate by -45 to +45 degrees
#         shear=(-3, 3),  # shear by -16 to +16 degrees
#         order=3,  # use any of scikit-image's interpolation methods
#         cval=(0.0, 1.0),  # if mode is constant, use a cval between 0 and 1.0
#         mode="constant"
#     )),
# ], random_order=True)
#
#
# class VisionLoader_AffChallenge2017(IDataLoader.IDataLoader):
#     numberOfAugmentedSamples = 10
#
#     @property
#     def logManager(self):
#         return self._logManager
#
#     @property
#     def dataTrain(self):
#         return self._dataTrain
#
#     @property
#     def dataValidation(self):
#         return self._dataValidation
#
#     @property
#     def dataTest(self):
#         return self._dataTest
#
#     @property
#     def preProcessingProperties(self):
#         return self._preProcessingProperties
#
#     def __init__(self, logManager, preProcessingProperties=None):
#
#         assert (not logManager == None), "No Log Manager was sent!"
#
#         self._preProcessingProperties = preProcessingProperties
#
#         self._dataTrain = None
#         self._dataTest = None
#         self._dataValidation = None
#         self._logManager = logManager
#
#     def dataAugmentation(self, dataPoint):
#
#         samples = []
#         samples.append(dataPoint)
#         for i in range(self.numberOfAugmentedSamples):
#             samples.append(seq.augment_image(dataPoint))
#
#         return numpy.array(samples)
#
#     def preProcess(self, dataLocation, augment=False):
#
#         assert (
#         not dataLocation == None or not self.preProcessingProperties == ""), "No data or parameter sent to preprocess!"
#
#         assert (len(self.preProcessingProperties[
#                         0]) == 2), "The preprocess have the wrong shape. Right shape: ([imgSize,imgSize])."
#
#
#         videoClip = VideoFileClip(dataLocation)
#
#
#
#
#         imageSize = self.preProcessingProperties[0]
#         grayScale = self.preProcessingProperties[1]
#
#
#         dataSequence = []
#         #print "Video:", dataLocation
#
#         for frame in videoClip.iter_frames():
#             data = numpy.array(cv2.resize(frame, imageSize))
#
#             if grayScale:
#                 data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
#
#             if augment:
#                 data = self.dataAugmentation(data)
#
#            # print "Shape:", numpy.array(data).shape
#
#             if grayScale:
#
#                 if augment:
#                     data = numpy.expand_dims(data, axis=1)
#                 else:
#                     data = numpy.expand_dims(data, axis=0)
#
#
#             else:
#
#                 if augment:
#                     data = numpy.swapaxes(data, 2, 3)
#                     data = numpy.swapaxes(data, 1, 2)
#                 else:
#                     data = numpy.swapaxes(data, 1, 2)
#                     data = numpy.swapaxes(data, 0, 1)
#
#             data = data.astype('float32')
#
#             data /= 255
#             dataSequence.append(data)
#
#         #print "clear videoClip"
#         videoClip.reader.close()
#         videoClip.audio.reader.close_proc()
#         videoClip.__del__()
#         dataSequence = numpy.array(dataSequence)
#
#         #print "Total Frames:", len(dataSequence)
#
#
#         return dataSequence
#
#     def orderClassesFolder(self, folder):
#
#         classes = os.listdir(folder)
#
#         return classes
#
#     def orderDataFolder(self, folder):
#
#         dataList = os.listdir(folder)
#         # print dataList
#         dataList = sorted(dataList, key=lambda x: int(str(x.split("_")[0])))
#
#         return dataList
#
#     def loadData(self, dataFolder, augment):
#
#         assert (not dataFolder == None or not dataFolder == ""), "Empty Data Folder!"
#
#         dataX = []
#         dataLabels = []
#         classesDictionary = []
#
#         videos = self.orderClassesFolder(dataFolder + "/")
#         self.logManager.write("Videos reading order: " + str(videos))
#         classNumber = 0
#
#         lastImage = None
#
#         numberOfVideos = 0
#         for v in videos:
#           numberOfVideos = numberOfVideos+1
#           if numberOfVideos < 3:
#
#             dataPointsPerVideos = self.orderDataFolder(dataFolder + "/" + v)
#             time = datetime.datetime.now()
#
#             for dataPointLocation in dataPointsPerVideos:
#
#                 dataSequence = self.preProcess(dataFolder + "/" + v + "/" + "/" + dataPointLocation, augment)
#
#                 # try:
#                 #     dataSequence = self.preProcess(dataFolder + "/" + v + "/" + "/" + dataPointLocation, augment)
#                 #
#                 #     if augment:
#                 #         lastImage = dataSequence[0]
#                 #     else:
#                 #         lastImage = dataSequence
#                 # except:
#                 #     print "Error!"
#                 #     if augment:
#                 #         dataSequence = [lastImage]
#                 #     else:
#                 #         dataSequence = lastImage
#
#
#
#                 #dataX.append(dataSequence) #For sequential data
#
#
#                 arousalLabel = dataPointLocation[0:-3].split("_")[1]
#                 #print "ArousalLabel:", arousalLabel
#                 valenceLabel = dataPointLocation[0:-3].split("_")[2][0:-1]
#                 #print "ValenceLabel:", valenceLabel
#                 #dataLabels.append(str(arousalLabel)+"_"+str(valenceLabel))
#
#                 dataLabels.append(float(arousalLabel)) # For sequential data
#
#                 for d in dataSequence: # for framewise data
#                     dataX.append(d)
#                     dataLabels.append(float(arousalLabel))
#
#
#                # print "DataSequences:", len(dataX)
#             print "DataSequences Shape:", numpy.array(dataX).shape
#             print "DataSequences Shape 0 :", numpy.array(dataX[0]).shape
#             print "DataSequences Shape 1 :", numpy.array(dataX[1]).shape
#                # raw_input("here11")
#
#             self.logManager.write(
#                 "--- Video: " + str(v) + "(" + str(len(dataPointsPerVideos)) + " Data points - " + str(
#                     (datetime.datetime.now() - time).total_seconds()) + " seconds" + ")")
#
#
#         # classNumber = classNumber+1
# #        dataLabels = np_utils.to_categorical(dataLabels, classNumber)
#
#         #        print dataLabels[0]
#         #        print numpy.shape(dataLabels)
#         #        raw_input("here")
#
#         dataX = numpy.array(dataX)
#         #        grayScale = self.preProcessingProperties[1]
#         #
#         #        if grayScale:
#         #            dataX = numpy.expand_dims(dataX,axis=1) #For grayscale
#         #        else:
#         #            dataX = numpy.swapaxes(dataX, 3,1) #"For color images"
#         #
#
#
#         dataX = dataX.astype('float32')
#
#         #        dataX /= 255
#
#         #        print "dataX shape:", numpy.array(dataX)
#         dataX, dataLabels = self.shuffleData(dataX, dataLabels)
#
#         #        print "dataX shape:", numpy.array(dataX)
#         #        raw_input("here")
#
#         dataX = numpy.array(dataX)
#         dataLabels = numpy.array(dataLabels)
#
#         return Data.Data(dataX, dataLabels, classesDictionary)
#
#     def loadTrainData(self, dataFolder, augmentData=False):
#         self.logManager.newLogSession("Training Data")
#         self.logManager.write("Loading From: " + dataFolder)
#         self.logManager.write("Preprocessing Properties: " + str(self.preProcessingProperties))
#         self._dataTrain = self.loadData(dataFolder, augmentData)
#         self.logManager.write("Total data points: " + str(len(self.dataTrain.dataX)))
#         self.logManager.write("Data points shape: " + str(numpy.array(self.dataTrain.dataX).shape))
#         self.logManager.write("Data labels shape: " + str(numpy.array(self.dataTrain.dataY).shape))
#         self.logManager.endLogSession()
#
#     def loadTestData(self, dataFolder, augmentData=False):
#         self.logManager.newLogSession("Testing Data")
#         self.logManager.write("Loading From: " + dataFolder)
#         self.logManager.write("Preprocessing Properties: " + str(self.preProcessingProperties))
#         self._dataTest = self.loadData(dataFolder, augmentData)
#         self.logManager.write("Total data points: " + str(len(self.dataTest.dataX)))
#         self.logManager.write("Data points shape: " + str(numpy.array(self.dataTest.dataX).shape))
#         self.logManager.write("Data labels shape: " + str(numpy.array(self.dataTest.dataY).shape))
#         self.logManager.endLogSession()
#
#     def loadValidationData(self, dataFolder, augmentData=False):
#         self.logManager.newLogSession("Validation Data")
#         self.logManager.write("Loading From: " + dataFolder)
#         self.logManager.write("Preprocessing Properties: " + str(self.preProcessingProperties))
#         self._dataValidation = self.loadData(dataFolder, augmentData)
#         self.logManager.write("Total data points: " + str(len(self.dataValidation.dataX)))
#         self.logManager.write("Data points shape: " + str(numpy.array(self.dataValidation.dataX).shape))
#         self.logManager.write("Data labels shape: " + str(numpy.array(self.dataValidation.dataY).shape))
#         self.logManager.endLogSession()
#
#     def saveData(self, folder):
#         pass
#
#     def shuffleData(self, dataX, dataY):
#
#         positions = []
#         for p in range(len(dataX)):
#             positions.append(p)
#
#         random.shuffle(positions)
#
#         newInputs = []
#         newOutputs = []
#         for p in positions:
#             newInputs.append(dataX[p])
#             newOutputs.append(dataY[p])
#
#         return (newInputs, newOutputs)
#
#     def loadTrainTestValidationData(self, folder, percentage):
#         pass
#
#     def loadNFoldValidationData(self, folder, NFold):
#         pass