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

from keras import regularizers

from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization

from keras.models import load_model

from keras.models import Model

import numpy
import copy

from keras import backend as K

K.set_image_dim_ordering('th')

# from keras.utils.layer_utils import print_layer_shapes


import IModelImplementation


class Audio_CNN_A5Experiment(IModelImplementation.IModelImplementation):
    batchSize = 16
    numberOfEpochs = 50

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

    def buildModel(self, inputShape, numberOfOutputs, name="AUDIO_SPec"):

        n_filters = 8
        hidden_size = 64
        n_layer = 2

        inputLayer = Input(shape=inputShape, name="Network_Input")

        # output: (N, 32, 32, n_filters)
        dx = self.convolution_image_for_encoding(inputLayer, n_filters, strides=(2, 2), name="%s/L1" % name,
                                                 n_layer=n_layer)

        # output: (N, 16, 16, n_filters*2)
        dx = self.convolution_image_for_encoding(dx, n_filters * 2, strides=(2, 2), name="%s/L2" % name,
                                                 n_layer=n_layer)

        # output: (N, 8, 8, n_filters*3)
#        dx = self.convolution_image_for_encoding(dx, n_filters * 3, strides=(2, 2), name="%s/L3" % name,
 #                                                n_layer=n_layer)

        # output: (N, 8, 8, n_filters*4)
  #      dx = self.convolution_image_for_encoding(dx, n_filters * 4, strides=(1, 1), name="%s/L4" % name,
   #                                              n_layer=n_layer)

        dx = Flatten()(dx)
        hidden = Dense(hidden_size, activation='relu', name="%s/Dense" % name)(dx)
        output = Dense(numberOfOutputs, activation="softmax")(hidden)
        model = Model(inputs=inputLayer, outputs=output)

        self._model = model

        # print_layer_shapes(self.model,input_shapes=inputShape)

        self.model.summary()

        if not self.logManager is None:
            self.logManager.endLogSession()

    def train(self, dataPointsTrain, dataPointsValidation, dataAugmentation):

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

        if not self.logManager is None:
            self.logManager.newLogSession("Training Model")

            # optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        #        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-5)
        optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        if not self.logManager is None:
            self.logManager.write("Training Strategy: " + str(optimizer.get_config()))

        self.model.compile(loss="categorical_crossentropy",
                           optimizer=optimizer,
                           metrics=['accuracy', 'categorical_accuracy', fbeta_score, recall, precision])

        filepath = self.experimentManager.modelDirectory + "/weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_categorical_accuracy', verbose=1, save_best_only=True,
                                     mode='max')
        early_stopping = EarlyStopping(monitor='val_loss', mode="min", patience=25)
        reduce_lr = ReduceLROnPlateau(factor=0.5, monitor='val_loss', min_lr=1e-5, patience=2)

        callbacks_list = [checkpoint, early_stopping, reduce_lr]

        if dataAugmentation:

            # this will do preprocessing and realtime data augmentation
            datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=True,  # apply ZCA whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)  # randomly flip images
            # compute quantities required for featurewise normalization
            # (std, mean, and principal components if ZCA whitening is applied)
            datagen.fit(dataPointsTrain.dataX)

            # fit the model on the batches generated by datagen.flow()
            history_callback = self.model.fit_generator(
                datagen.flow(dataPointsTrain.dataX, dataPointsTrain.dataY, shuffle=True,
                             batch_size=self.batchSize),
                steps_per_epoch=dataPointsTrain.dataX.shape[0] / self.batchSize,
                epochs=self.numberOfEpochs,
                validation_data=(dataPointsValidation.dataX, dataPointsValidation.dataY),
                callbacks=callbacks_list)


        else:
            history_callback = self.model.fit(dataPointsTrain.dataX, dataPointsTrain.dataY,
                                              batch_size=self.batchSize,
                                              epochs=self.numberOfEpochs,
                                              validation_data=(dataPointsValidation.dataX, dataPointsValidation.dataY),
                                              shuffle=True,
                                              callbacks=callbacks_list)

            if not self.logManager is None:
                self.logManager.write(str(history_callback.history))
                self.logManager.endLogSession()

            if not self.plotManager is None:
                self.plotManager.createTrainingPlot(history_callback, self.modelName)

            self.model.load_weights(self.experimentManager.modelDirectory + "/weights.best.hdf5")

            self.model.compile(loss='categorical_crossentropy',
                               optimizer=optimizer,
                               metrics=['accuracy', 'categorical_accuracy', fbeta_score, recall, precision])

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
