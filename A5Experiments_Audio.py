# -*- coding: utf-8 -*-

import matplotlib

matplotlib.use('Agg')

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras import backend as K


def set_keras_backend(backend):
    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend


def runModel():
    from KEF.Controllers import ExperimentManager

    # from KEF.DataLoaders import FER2013PlusLoader
    from KEF.DataLoaders import AudioLoader_Spectrogram_A5Experiment

    from KEF.Implementations import Audio_CNN_A5Experiment

    dataDirectory = "/data/experiments_A5_Sound_1Phoneme_EgoNoise/"

    datasetFolderTrain = "/data/datasets/A5_Experiments/RecordedSounds_A5Experiments_Train_1_Phoneme_EgoNoise/train/"


    datasetFolderTest = "/data/datasets/A5_Experiments/RecordedSounds_A5Experiments_Train_1_Phoneme_EgoNoise/test/"



    """ Initianize all the parameters and modules necessary

         image size: 64,64
    """

    experimentManager = ExperimentManager.ExperimentManager(dataDirectory,
                                                            "AudioSet_Audio_DeepNetwork_Frame_01",
                                                            verbose=True)

    grayScale = True

    preProcessingProperties = [(64, 64), grayScale]

    fps = 30
    stride = 30

    """ Loading the training and testing data 

    """

    dataLoader = AudioLoader_Spectrogram_A5Experiment.AudioLoader_Spectrogram_A5Experiment(experimentManager.logManager, preProcessingProperties)
    #


    #
    #raw_input("here")
    dataLoader.loadTrainData(datasetFolderTrain, augmentData=False)

    dataLoader.loadTestData(datasetFolderTest, augmentData=False)
    #


    #raw_input("here")
    # dataLoader.loadValidationData(datasetFolderValidation)


    # """ Creating and tuning the CNN
    # """


    cnnModel = Audio_CNN_A5Experiment.Audio_CNN_A5Experiment(experimentManager, "Audio_Deep_CNN", experimentManager.plotManager)


    #
    cnnModel.buildModel(dataLoader.dataTest.dataX.shape[1:], len(dataLoader.dataTest.labelDictionary))
    ##
    cnnModel.train(dataLoader.dataTrain, dataLoader.dataTest, False)
    ##

    cnnModel.save(experimentManager.modelDirectory)
    ##


    print "Private Test Evaluation"
    #cnnModel.evaluate(dataLoader.dataTest)


set_keras_backend("tensorflow")

print K.backend

if K.backend == "tensorflow":
    import tensorflow as tf



    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    # from keras import backend as K
    K.set_session(sess)

    with tf.device('/gpu:0'):
        runModel()
else:

    runModel()