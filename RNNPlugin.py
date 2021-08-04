#import tensorflow as tf
#from tensorflow.keras import datasets, layers, models
#import matplotlib.pyplot as plt
import numpy
#import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from tensorflow import keras
from tensorflow.keras import layers

from PIL import Image as im

import PyPluMA

class RNNPlugin:
    def input(self, filename):
       self.parameters = dict()
       paramfile = open(filename, 'r')
       for line in paramfile:
         contents = line.split('\t')
         self.parameters[contents[0]] = contents[1].strip()



       classnames_file = open(PyPluMA.prefix()+"/"+self.parameters['classnames'], 'r')
       self.class_names = []
       for line in classnames_file:
           self.class_names.append(line.strip())


       train_file = open(PyPluMA.prefix()+"/"+self.parameters['trainset'], 'r')
       pos = 0
       train_image_list = []
       train_label_list = []
       for line in train_file:
           line = line.strip()
           contents = line.split(',')
           data = numpy.asarray(im.open(PyPluMA.prefix()+"/"+contents[0]))
           data = data.reshape([32,96])
           print("READING FILE "+contents[0])
           train_image_list.append([data])
           train_label_list.append(numpy.asarray([[self.class_names.index(contents[1])]]))
           pos += 1
       train_images = numpy.vstack(tuple(train_image_list))
       train_labels = numpy.vstack(tuple(train_label_list))
       #print(train_images.shape)
       #exit()

       self.inputfilenames = []
       test_file = open(PyPluMA.prefix()+"/"+self.parameters['testset'], 'r')
       pos = 0
       test_image_list = []
       test_label_list = []
       for line in test_file:
           line = line.strip()
           contents = line.split(',')
           data = numpy.asarray(im.open(PyPluMA.prefix()+"/"+contents[0]))
           data = data.reshape([32,96])
           self.inputfilenames.append(contents[0])
           print("READING FILE "+contents[0])
           test_image_list.append([data])
           test_label_list.append(numpy.asarray([[self.class_names.index(contents[1])]]))
           pos += 1
       self.test_images = numpy.vstack(tuple(test_image_list))
       self.test_labels = numpy.vstack(tuple(test_label_list))

       # Normalize pixel values to be between 0 and 1
       train_images, self.test_images = train_images / 255.0, self.test_images / 255.0
       #batch_size = 64

       #units = 64

       lstm_layer = keras.layers.RNN(
            keras.layers.LSTMCell(len(self.class_names)), input_shape=(len(train_images[0]), len(train_images[0][0])))
       self.model = keras.models.Sequential(
       [
            lstm_layer,
            keras.layers.BatchNormalization(),
            keras.layers.Dense(len(self.class_names)),
       ]
       )
       self.model.compile(
       loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
       optimizer="sgd",
       metrics=["accuracy"],
       )
       self.model.fit(
          train_images, train_labels, validation_data=(self.test_images, self.test_labels), batch_size=64, epochs=1
       )


        

    def run(self):
       #self.test_loss, self.test_acc = self.model.evaluate(self.test_images, self.test_labels, verbose=2)
       probability_model = tf.keras.Sequential([self.model,
                                         tf.keras.layers.Softmax()])
       self.predictions = probability_model.predict(self.test_images)

    def output(self, filename):
       outfile = open(filename+".probs.csv", 'w')
       outfile2 = open(filename+".final.csv", 'w')
       #print(self.test_loss)
       #print(self.test_acc)
       outfile.write("\"\",")
       for i in range(0, len(self.class_names)):
           outfile.write(self.class_names[i])
           if (i != len(self.class_names)-1):
               outfile.write(',')
           else:
               outfile.write('\n')
       for i in range(0, len(self.predictions)):
           outfile.write(self.inputfilenames[i]+",")
           outfile2.write(self.inputfilenames[i]+",")
           maxval = -1
           maxindex = -1
           for j in range(0, len(self.predictions[i])):
               outfile.write(str(self.predictions[i][j]))
               if (self.predictions[i][j] > maxval):
                   maxval = self.predictions[i][j]
                   maxindex = j
               if (j != len(self.predictions[i])-1):
                   outfile.write(',')
               else:
                   outfile.write('\n')
           outfile2.write(self.class_names[maxindex]+"\n")
