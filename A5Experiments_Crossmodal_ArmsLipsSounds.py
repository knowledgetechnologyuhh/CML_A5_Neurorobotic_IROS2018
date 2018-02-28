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
    from KEF.DataLoaders import CrossmodalLoader_A5Experiments_LipsArms_Framebased_SoundSpectogram

    from KEF.Implementations import Crossmodal_CNN_A5Experiment



    #Data Location
    datasetArmsFolder = "/data/datasets/A5_Experiments/A5VideoExperiments_separated/Arms_Average/all/"
    datasetLipsFolder = "/data/datasets/A5_Experiments/A5VideoExperiments_separated/Lips_Face_Average/all/"
    datasetSoundsFolder = "/data/datasets/A5_Experiments/RecordedSound_A5Experiments/"

    datasetParticipantsFolder = "//data/datasets/A5_Experiments/A5_original data_33participants/"

    grayScale = True
    preProcessingProperties = [(120, 120), grayScale]



    numberOfParticipants = range(33)
    #numberOfParticipants = range(2)

    for participantNumber in numberOfParticipants:
        trainData = list(numberOfParticipants)
        trainData.remove(participantNumber)
        from random import shuffle

        shuffle(trainData)
        trainData = trainData[0:5]

        dataDirectory = "/data/experiments_A5_Crossmodal_Arms_Lips_Sound_Without_Feedback" + "/Participant_"+str(participantNumber)

        experimentManager = ExperimentManager.ExperimentManager(dataDirectory,
                                                                 "AudioSet_Audio_DeepNetwork_Frame_01",
                                                                 verbose=True)

        dataLoader = CrossmodalLoader_A5Experiments_LipsArms_Framebased_SoundSpectogram.CrossmodalLoader_A5Experiments_LipsArms_Framebased_SoundSpectogram(
            experimentManager.logManager, preProcessingProperties)

        #print "Train Data:", trainData
        #print "Number of participants:", numberOfParticipants
        #print "Participant Number:", participantNumber
        #dataLoader.loadTrainData(datasetLipsFolder,datasetArmsFolder,datasetSoundsFolder,datasetParticipantsFolder,trainData)
        testCSV = dataLoader.loadTestData(datasetLipsFolder,datasetArmsFolder,datasetSoundsFolder,datasetParticipantsFolder,[participantNumber])

        print "Test CSV:", testCSV
        cnnModel = Crossmodal_CNN_A5Experiment.Crossmodal_CNN_A5Experiment(experimentManager, "Crossmodal_Deep_CNN", experimentManager.plotManager)

        cnnModel.buildModel()

        cnnModel.train(dataLoader.dataTest, dataLoader.dataTest,False)

        # participantsCSVs = os.listdir(datasetParticipantsFolder)
        # testCSV = ""
        # for participantCSV in participantsCSVs:
        #     if participantNumber in [participantNumber]:
        #         testCSV = participantCSV

        #self, dataLoader, participantData, subjectId, savingCSV):


        #validationCSV = dataLoader.loadValidationData(datasetLipsFolder,datasetArmsFolder,datasetSoundsFolder,datasetParticipantsFolder,[participantNumber])
        cnnModel.getResponses(dataLoader.dataTest,testCSV, participantNumber, "/data/datasets/A5_Experiments/CSV_Outputs_With_Feedback/Participant_"+str(participantNumber)+".csv")

        #raw_input("here")




    # """ Initianize all the parameters and modules necessary
    #
    #      image size: 64,64
    # """
    #
    # experimentManager = ExperimentManager.ExperimentManager(dataDirectory,
    #                                                         "AudioSet_Audio_DeepNetwork_Frame_01",
    #                                                         verbose=True)
    #
    # grayScale = True
    # #grayScale = False # WORKING1
    #
    # #preProcessingProperties = [(320, 240), grayScale]
    # preProcessingProperties = [(120, 120), grayScale]
    # #preProcessingProperties = [(160, 120), grayScale]
    #
    # #preProcessingProperties = [(128, 128), grayScale]
    #
    # fps = 30
    # stride = 30
    #
    # """ Loading the training and testing data
    #
    # """
    #
    # dataLoader = VisionLoader_A5Experiments_LipsFrameBased.VisionLoader_A5Experiments_LipsFrameBased(experimentManager.logManager, preProcessingProperties)
    # #
    #
    #
    # #
    # #raw_input("here")
    # dataLoader.loadTrainData(datasetFolderTrain, augmentData=False)
    #
    # dataLoader.loadTestData(datasetFolderTest, augmentData=False)
    # #
    #
    #
    # #raw_input("here")
    # # dataLoader.loadValidationData(datasetFolderValidation)
    #
    #
    # # """ Creating and tuning the CNN
    # # """
    #
    #
    # cnnModel = Vision_CNN_A5Experiment_Frame_Lips.Vision_CNN_A5Experiment_Frame_Lips(experimentManager, "Vision_Deep_CNN", experimentManager.plotManager)
    #
    #
    # #
    # cnnModel.buildModel(dataLoader.dataTest.dataX.shape[1:], len(dataLoader.dataTest.labelDictionary))
    # ##
    # cnnModel.train(dataLoader.dataTrain, dataLoader.dataTest, False)
    # ##
    #
    # cnnModel.save(experimentManager.modelDirectory)
    # ##
    #
    #
    # print "Private Test Evaluation"
    # #cnnModel.evaluate(dataLoader.dataTest)


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