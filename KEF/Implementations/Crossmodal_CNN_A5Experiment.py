# -*- coding: utf-8 -*-
import matplotlib

matplotlib.use('Agg')

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Input
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import LeakyReLU
from keras.optimizers import Adam, Adamax
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import r2_score

from keras.layers import GlobalAveragePooling1D, concatenate, add

from keras import regularizers

from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization

from keras.models import load_model

from keras.models import Model

import datetime

import numpy
import copy

from keras import backend as K

K.set_image_dim_ordering('th')

from KEF.Metrics import metrics

# from keras.utils.layer_utils import print_layer_shapes


import IModelImplementation


class Crossmodal_CNN_A5Experiment(IModelImplementation.IModelImplementation):
    batchSize = 32
    numberOfEpochs = 100

    @property
    def modelName(self):
        return self._modelName

    @property
    def model(self):
        return self._model

    @property
    def logManager(self):
        return self._logManager

    @property
    def experimentManager(self):
        return self._experimentManager

    @property
    def plotManager(self):
        return self._plotManager

    def r2_score(self, y_true, y_pred):
        """Implements r2_score metric from sklearn"""
        return r2_score(y_true, y_pred, multioutput="raw_values")

    def hinge_onehot(self, y_true, y_pred):
        y_true = y_true * 2 - 1
        y_pred = y_pred * 2 - 1

        return K.mean(K.maximum(1. - y_true * y_pred, 0.), axis=-1)

    def __init__(self, experimentManager=None, modelName=None, plotManager=None):
        self._logManager = experimentManager.logManager
        self._experimentManager = experimentManager
        self._modelName = modelName
        self._plotManager = plotManager

    def convolution_image_for_encoding(self, x, filters, strides=(1, 1), name=None, n_layer=2):
        for i in range(1, n_layer):
            x = Conv2D(filters, (3, 3), activation="elu", padding="same", name="%s/Conv%d" % (name, i))(x)

        x = Conv2D(filters, (3, 3), activation="elu", padding="same", strides=strides,
                   name="%s/Conv%d" % (name, n_layer))(x)
        return x


    def loadAudioModel(self, modelLocation):

        audioModel = load_model(modelLocation,
                                custom_objects={'fbeta_score': metrics.fbeta_score, 'recall': metrics.recall, 'precision': metrics.precision})

        audioModel.summary()
        for layer in audioModel.layers:
            layer.name = "Audio_" + layer.name




        print "Full AudioModel"
        audioModel.summary()

        #model = Model(inputs=audioModel.input, outputs=audioModel.get_layer(name="flatten_1").output)



        return audioModel.input, audioModel.get_layer(name="Audio_dense_1").output



    def loadLipsModel(self, modelLocation):

        lipsModel = load_model(modelLocation, custom_objects={'fbeta_score': metrics.fbeta_score, 'recall': metrics.recall, 'precision': metrics.precision})

        #lipsModel.summary()
        for layer in lipsModel.layers:
            layer.name = "Lips_" + layer.name

        #lipsModel.get_layer(name='Network_Input').name = 'Lips_Network_Input'  # "model_1 : rename as you like"
        print "Full Lips Model"
        lipsModel.summary()


        #model = Model(inputs=lipsModel.input, outputs=lipsModel.get_layer(name="flatten_1").output)



        return lipsModel.input, lipsModel.get_layer(name="Lips_dense_1").output

    def loadArmsModel(self, modelLocation):

        armsModel = load_model(modelLocation, custom_objects={'fbeta_score': metrics.fbeta_score, 'recall': metrics.recall, 'precision': metrics.precision})

        #armsModel.summary()
        for layer in armsModel.layers:
            layer.name = "Arms_" + layer.name


        print "Full Arms Model"
        armsModel.summary()

        #model = Model(inputs=armsModel.input, outputs=armsModel.get_layer(name="flatten_1").output)

        return armsModel.input, armsModel.get_layer(name="Arms_dense_1").output



    def buildModel(self):

        audioModelDirectory = "//data/datasets/A5_Experiments/trainedNetworks/Audio_1Phoneme_No_EgoNoise/Model/weights.best.hdf5"

        lipsModelDirectory = "/data/datasets/A5_Experiments/trainedNetworks/Lips_Average/Model/weights.best.hdf5"

        armsModelDirectory = "/data/datasets/A5_Experiments/trainedNetworks/Arms_Average/Model/weights.best.hdf5"

        inputAudio, audioModel = self.loadAudioModel(audioModelDirectory)

        inputLips, lipsModel = self.loadLipsModel(lipsModelDirectory)

        inputArms, armsmodel = self.loadArmsModel(armsModelDirectory)


        concatenation = concatenate([armsmodel, lipsModel, audioModel])

        dense = Dense(200, activation="relu", name="crossmodalDense")(concatenation)
        dropDense = Dropout(0.25)(dense)

        crossmodalModelOutput = Dense(units=4, activation="softmax", name="Cross_Output")(dropDense)

        self._model = Model(inputs=[inputArms, inputLips, inputAudio], outputs=crossmodalModelOutput)

        for layer in self.model.layers:
            layer.trainable = False


        self.model.get_layer(name="crossmodalDense").trainable = True
        self.model.get_layer(name="Cross_Output").trainable = True

        self.model.summary()

        self.plotManager.creatModelPlot(self.model, str(self.modelName))

        #raw_input("here")

        if not self.logManager is None:
            self.logManager.endLogSession()

    def train(self, dataPointsTrain, dataPointsValidation, dataAugmentation):


        if not self.logManager is None:
            self.logManager.newLogSession("Training Model")

            # optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        #        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-5)
        optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        if not self.logManager is None:
            self.logManager.write("Training Strategy: " + str(optimizer.get_config()))

        self.model.compile(loss="categorical_crossentropy",
                           optimizer=optimizer,
                           metrics=['accuracy', metrics.fbeta_score, metrics.recall, metrics.precision])

        filepath = self.experimentManager.modelDirectory + "/weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_categorical_accuracy', verbose=1, save_best_only=True,
                                     mode='max')
        early_stopping = EarlyStopping(monitor='val_loss', mode="min", patience=25)
        reduce_lr = ReduceLROnPlateau(factor=0.5, monitor='val_loss', min_lr=1e-5, patience=2)

        callbacks_list = [checkpoint, early_stopping, reduce_lr]

        #concatenation = concatenate([armsmodel, lipsModel, audioModel])
        #audioModel = dataPointsTrain.dataX[:, 0]

        #print "Shape1:", numpy.array(dataPointsTrain.dataX[:, 0]).shape

        armsTrain = []
        lipsTrain = []
        audioTrain = []
        for i in dataPointsTrain.dataX:
            armsTrain.append(i[0])
            lipsTrain.append(i[1])
            audioTrain.append(i[2])
        armsTrain = numpy.array(armsTrain)
        lipsTrain = numpy.array(lipsTrain)
        audioTrain = numpy.array(audioTrain)

        armsValidation = []
        lipsValidation = []
        audioValidation = []
        for i in dataPointsValidation.dataX:
            armsValidation.append(i[0])
            lipsValidation.append(i[1])
            audioValidation.append(i[2])
        armsValidation = numpy.array(armsValidation)
        lipsValidation = numpy.array(lipsValidation)
        audioValidation = numpy.array(audioValidation)


        history_callback = self.model.fit([armsTrain,lipsTrain, audioTrain] , dataPointsTrain.dataY,
                                          batch_size=self.batchSize,
                                          epochs=self.numberOfEpochs,
                                          validation_data=([armsValidation,lipsValidation, audioValidation], dataPointsValidation.dataY),
                                          shuffle=True,
                                          callbacks=callbacks_list)

        if not self.logManager is None:
            self.logManager.write(str(history_callback.history))
            self.logManager.endLogSession()

        if not self.plotManager is None:
            self.plotManager.createTrainingPlot(history_callback, self.modelName)


    def getOutputFromConvLayer(self, data, layerName):

        # self.model.summary()
        somData = copy.deepcopy(data)

        dataX = []

        for i in range(len(data.dataX)):
            intermediate_layer_model = Model(inputs=self.model.input, outputs=self.model.get_layer(layerName).output)
            output = numpy.array(intermediate_layer_model.predict(numpy.array([data.dataX[i]]))).flatten()

            dataX.append(output)

        dataX = numpy.array(dataX)
        somData.dataX = dataX
        return somData

    def getResponses(self, dataLoader, participantData, subjectId, savingCSV):

        import csv

        print "Loading From:", participantData

        with open(participantData, 'rb') as csvfile:
            with open(savingCSV, 'wb') as outputCSVfile:

                reader = csv.reader(csvfile)
                csvwriter = csv.writer(outputCSVfile, delimiter=',')

                csvwriter.writerow(["#","SubjectID","Age","Gender","Block","TrialID","Audio","Lips","Arm","Congruency","Condition","ResponseTime","Accuracy","Sound","Decision"])
                rownum = 0
                dataPointNumber = 0
                for row in reader:
                    # print "Row Number:", rownum
                    if rownum >= 1:
                        #print "Row:", row
                        audioLabel = row[6]
                        lipsLabel = row[7]
                        armsLabel = row[8]
                        congruencyLabel = row[9]
                        conditionLabel = row[10]
                        phonemeLabel = row[-2]
                        decisionLabel = row[-1]

                        if decisionLabel == "Na":
                            newRow = [row[0], "rob_"+str(subjectId), "0", "robot", row[4], row[5], row[6], row[7], row[8], row[9],
                                      row[10], "Na", "0", row[13], "Na"]
                        else:
                            armInput = numpy.array([dataLoader.dataX[dataPointNumber][0]])
                            lipsInput = numpy.array([dataLoader.dataX[dataPointNumber][1]])
                            audioInput = numpy.array([dataLoader.dataX[dataPointNumber][2]])
                            labelInput = numpy.array(dataLoader.dataY[dataPointNumber])

                            # print "Shape Arms:", armInput.shape
                            # print "Shape Lips:", lipsInput.shape
                            # print "Shape Sound:", audioInput.shape

                            time = datetime.datetime.now()
                            prediction = self.model.predict([armInput,lipsInput,audioInput], batch_size=self.batchSize)

                            # print "DataPoint Number: ", dataPointNumber
                            # print "Shape:", dataLoader.dataY.shape
                            # print "Shape0:", dataLoader.dataY[dataPointNumber].shape
                            # print "Label Input:", dataLoader.dataY[dataPointNumber]
                            # print "Label Argmax:", numpy.argmax( dataLoader.dataY[dataPointNumber]) +1
                            # print "Audio Label:", audioLabel
                            # print "Prediction Label:", prediction
                            # print "Argmax Prediction Label:", numpy.argmax(prediction) +1
                            # print "Decision Label:", decisionLabel
                            #
                            #
                            # print "------------------------"
                            # raw_input("here")

                            prediction = numpy.argmax(prediction) + 1

                            if int(prediction) == int(audioLabel):
                                accuracy = str(1)
                            else:
                                accuracy = str(0)

                            newRow = [row[0], "rob_"+str(subjectId), "0", "robot", row[4], row[5], row[6], row[7], row[8], row[9],
                                      row[10], (datetime.datetime.now()-time).total_seconds()*1000, accuracy, row[13], prediction]

                            dataPointNumber = dataPointNumber + 1
                        csvwriter.writerow(newRow)

                    rownum = rownum+1
                    # if rownum == 30:
                    #     break





    def evaluate(self, dataPoints):
        if not self.logManager is None:
            self.logManager.newLogSession("Model Evaluation")

        evaluation = self.model.evaluate(dataPoints.dataX, dataPoints.dataY, batch_size=self.batchSize)

        if not self.logManager is None:
            self.logManager.write(str(evaluation))
            self.logManager.endLogSession()

        return evaluation

    def classify(self, dataPoint):
        # Todo
        return self.model.predict_classes(numpy.array([dataPoint]), batch_size=self.batchSize, verbose=0)

    def save(self, saveFolder):

        print "Save Folder:", saveFolder + "/" + self.modelName + ".h5"
        self.model.save(saveFolder + "/" + self.modelName + ".h5")

    def load(self, loadFolder):
        def precision(y_true, y_pred):
            from keras import backend as Kend
            """Precision metric.
            Only computes a batch-wise average of precision.
            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
            true_positives = Kend.sum(Kend.round(Kend.clip(y_true * y_pred, 0, 1)))
            predicted_positives = Kend.sum(Kend.round(Kend.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + Kend.epsilon())
            return precision

        def recall(y_true, y_pred):
            from keras import backend as Kend
            """Recall metric.
            Only computes a batch-wise average of recall.
            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            """
            true_positives = Kend.sum(Kend.round(Kend.clip(y_true * y_pred, 0, 1)))
            possible_positives = Kend.sum(Kend.round(Kend.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + Kend.epsilon())
            return recall

        def fbeta_score(y_true, y_pred, beta=0.5):
            from keras import backend as Kend
            """Computes the F score.
            The F score is the weighted harmonic mean of precision and recall.
            Here it is only computed as a batch-wise average, not globally.
            This is useful for multi-label classification, where input samples can be
            classified as sets of labels. By only using accuracy (precision) a model
            would achieve a perfect score by simply assigning every class to every
            input. In order to avoid this, a metric should penalize incorrect class
            assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
            computes this, as a weighted mean of the proportion of correct class
            assignments vs. the proportion of incorrect class assignments.
            With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
            correct classes becomes more important, and with beta > 1 the metric is
            instead weighted towards penalizing incorrect class assignments.
            """
            if beta < 0:
                raise ValueError('The lowest choosable beta is zero (only precision).')

            # If there are no true positives, fix the F score at 0 like sklearn.
            if Kend.sum(Kend.round(Kend.clip(y_true, 0, 1))) == 0:
                return 0

            p = precision(y_true, y_pred)
            r = recall(y_true, y_pred)
            bb = beta ** 2
            fbeta_score = (1 + bb) * (p * r) / (bb * p + r + Kend.epsilon())
            return fbeta_score

        def fmeasure(y_true, y_pred):
            """Computes the f-measure, the harmonic mean of precision and recall.
            Here it is only computed as a batch-wise average, not globally.
            """
            return fbeta_score(y_true, y_pred, beta=1)

        self._model = load_model(loadFolder,
                                 custom_objects={'fbeta_score': fbeta_score, 'recall': recall, 'precision': precision})
        self._model.summary()
