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
    from KEF.DataLoaders import VisionLoader_A5Experiments_LipsFrameBased

    from KEF.Implementations import Vision_CNN_A5Experiment_Frame_Lips

    dataDirectory = "/data/experiments_A5_Vision_FrameBased_Lips/"

    datasetFolderTrain = "/data/datasets/A5_Experiments/A5VideoExperiments_separated/Lips_Face_Average/train/"


    datasetFolderTest = "/data/datasets/A5_Experiments/A5VideoExperiments_separated/Lips_Face_Average/test/"
    #datasetFolderTest = "/data/datasets/A5_Experiments/A5VideoExperiments_separated/Arms_Average/train/"



    """ Initianize all the parameters and modules necessary

         image size: 64,64
    """

    experimentManager = ExperimentManager.ExperimentManager(dataDirectory,
                                                            "AudioSet_Audio_DeepNetwork_Frame_01",
                                                            verbose=True)

    grayScale = True
    #grayScale = False # WORKING1

    #preProcessingProperties = [(320, 240), grayScale]
    preProcessingProperties = [(120, 120), grayScale]
    #preProcessingProperties = [(160, 120), grayScale]

    #preProcessingProperties = [(128, 128), grayScale]

    fps = 30
    stride = 30

    """ Loading the training and testing data 

    """

    dataLoader = VisionLoader_A5Experiments_LipsFrameBased.VisionLoader_A5Experiments_LipsFrameBased(experimentManager.logManager, preProcessingProperties)
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


    cnnModel = Vision_CNN_A5Experiment_Frame_Lips.Vision_CNN_A5Experiment_Frame_Lips(experimentManager, "Vision_Deep_CNN", experimentManager.plotManager)


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