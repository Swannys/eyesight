# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 13:55:37 2024

@author: Brendan
"""

import tensorflow as tf
#import keras as keras
from tensorflow import device
from keras import backend as K
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Activation, BatchNormalization, Input, Layer
from tensorflow.keras.optimizers import Nadam, Adam, SGD
from tensorflow.keras.initializers import RandomNormal
from tensorflow.python.keras.utils import conv_utils
from keras.activations import relu
#from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime
from joblib import load, dump
import os
import gc

#'''
#Data_Directory = "C:\Users\Brendan\Downloads\Brendan_2023_24\Brendan_2023_24\Dataset\Preprocessed\Dataset"
#Categories = ["Bus", "Car", "Motorcycles", "Person"]
#training_data = []
#'''

class Fully_Connected_Scratch(Layer):
    def __init__(self, units, activation=relu, trainable = True):
        super(Fully_Connected_Scratch, self).__init__()
        self.units = units
        self.trainable= trainable
        self.activation = tf.keras.activations.get(activation)
        
    def build(self, input_shape):
        
        #instantiates weight values based on the number of connections using normal distribution
        self.w = self.add_weight(name = 'kernal',
                                 shape = (input_shape[1], self.units),
                                 initializer = 'random_normal',
                                 trainable = True)
        #instantiates bias values based on numer of nodes / zeros
        self.b = self.add_weight(name = 'bias',
                                 shape = (self.units,),
                                 initializer = 'zeros',
                                 trainable = True)
        super(Fully_Connected_Scratch, self).build(input_shape)
        
        def call(self, inputs):
            # tf.matmul computes dot product of two matricies
            return self.activation(tf.matmul(inputs, self.w) + self.b)
        
        def compute_output_shape(self, input_shape):
            return (input_shape[0], self.units)
        
        def get_config(self):
            return {"units": self.units}
        
        
        
class Conv2D_Scratch(Layer):
    def __init__(self, filters, kernel_size, strides = 1, dilation_rate = 1, padding = 'valid'):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = conv_utils.normalize_padding(padding)
        self.dilation_rate = dilation_rate
        super(Conv2D_Scratch, self).__init__()
        value = K.image_data_format()
        self.data_format = value.lower()
        
    def build(self, input_shape):
        shape = self.kernel_size + (input_shape[-1], self.filters)
        self.kernel = self.add_weight(name = 'kernel', shape = shape, initializer = 'random_normal')
        self.bias = self.add_weight(name = 'bias', shape = (self.filters,), initializer = 'zeros')
        super(Conv2D_Scratch, self).build(input_shape)
        
    def call(self, inputs):
        out_height = inputs.shape[1] - self.kernel_size[0] + 1
        out_width = inputs.shape[2] - self.kernel_size[1] + 1
        
        inputs = tf.reshape(inputs, [-1, inputs.shape[1], inputs.shape[2], inputs.shape[3]])
        
        patches = tf.image.extract_patches(inputs, [1, self.kernel_size[0], self.kernel_size[1], 1], [1,1,1,1], [1,1,1,1], 'VALID')
        patches = tf.reshape(patches, [-1, self.kernel_size[0] * self.kernel_size[1] * inputs.shape[3]])
        
        kernel = tf.reshape(self.kernel, [self.kernel_size[0] * self.kernel_size[1] * inputs.shape[3], self.filters])
        
        output = tf.matmul(patches, kernel)
        output = tf.reshape(output, [-1, out_height, out_width, self.filters])
        return output
    
    def get_config(self):
        config = super(Conv2D_Scratch, self).get_config()
        config.update({'filters':self.filters, 'kernel_size':self.kernel_size, 'strides':self.strides})
        return config
    
    
    
class Maxpool_Scratch(Layer):
    def __init__(self, pool_size=(2,2), strides=2, padding="VALID", trainable=False, data_format=None, **kwargs):
        super(Maxpool_Scratch, self).__init__()
        self.pool_size = conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = data_format
        self.trainable = trainable
        
    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]
            
        rows = conv_utils.conv_output_length(rows, self.pool_size[0], self.padding, self.strides[0])
        cols = conv_utils.output_length(cols, self.pool_size[1], self.padding, self.strides[1])
        
        if self.data_format == 'channels_first':
            return (input_shape[0], input_shape[1], rows, cols)
        elif self.data_format == 'channels_last':
            return (input_shape[0], rows, cols, input_shape[3])
        
    def call(self, inputs):
        patches = tf.image.extract_patches(inputs, sizes = [1, self.pool_size[0], self.pool_size[1], 1],
                                                   strides = [1, self.strides[0], self.strides[1], 1],
                                                   rates = [1,1,1,1],
                                                   padding = "VALID")
        
        shape = tf.shape(patches)
        patches = tf.reshape(patches, [shape[0], shape[1], shape[2], -1, shape[3]])
        
        output = tf.math.reduce_max(patches, axis = 3)
        return output
    
    def get_config(self):
        config = super(Maxpool_Scratch, self).get_config()
        config.update({'pool_size':self.pool_size, 'strides':self.strides, 'padding':self.padding})
        return config
    


class EyesightManager():
    def __init__(self, name='Eyesight_VGG16', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4)):
        self.name = name
        self.model = None
        self.optimizer = optimizer
        self.Image_train = None
        self.Image_test = None
        self.Label_train = None
        self.Label_test = None
        
    def rename_model(self, new_name):
        self.name = new_name
        
    def build_eyesight(self):
        
        self.model = Sequential()
        
        print('First Conv Layer 64 filter of size 3x3')
        self.model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same', 
                 input_shape = (244, 244, 3), data_format = "channels_last"))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size = (2, 2), padding = 'same')) #MaxPooling of size 2x2
         
        print('Second Conv Layer 128 filter of size 3x3')
        self.model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding = 'same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding = 'same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size = (2, 2), padding = 'same')) #MaxPooling of size 2x2
         
        print('Third Conv Layer 256 filter of size 3x3')
        self.model.add(Conv2D(filters = 256, kernel_size = (3, 3), padding = 'same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(Conv2D(filters = 256, kernel_size = (3, 3), padding = 'same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(Conv2D(filters = 256, kernel_size = (3, 3), padding = 'same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size = (2, 2), padding = 'same')) #MaxPooling of size 2x2
         
        print('Fourth Conv Layer 512 filter of size 3x3')
        self.model.add(Conv2D(filters = 512, kernel_size = (3, 3), padding = 'same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(Conv2D(filters = 512, kernel_size = (3, 3), padding = 'same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(Conv2D(filters = 512, kernel_size = (3, 3), padding = 'same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size = (2, 2), padding = 'same')) #MaxPooling of size 2x2
         
        print('Fifth Conv Layer 512 filter of size 3x3')
        self.model.add(Conv2D(filters = 512, kernel_size = (3, 3), padding = 'same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(Conv2D(filters = 512, kernel_size = (3, 3), padding = 'same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(Conv2D(filters = 512, kernel_size = (3, 3), padding = 'same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu", name = 'last_layer'))
        self.model.add(MaxPooling2D(pool_size = (2, 2), padding = 'same')) #MaxPooling of size 2x2
         
        print('1st FC with 512 Neurons & ReLu activation')
        self.model.add(Flatten())
        self.model.add(Dense(512, kernel_initializer = RandomNormal(mean = 0.0, stddev = 0.01, seed = None)))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.5))
         
        print('2nd FC with 512 Neurons & ReLu activation')
        self.model.add(Dense(512, kernel_initializer = RandomNormal(mean = 0.0, stddev = 0.01, seed = None)))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.5))
         
        print('3nd FC with 128 Neurons & ReLu activation')
        self.model.add(Dense(128, kernel_initializer = RandomNormal(mean = 0.0, stddev = 0.01, seed = None)))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.5))
         
        print('FC with 4 Neurons & SoftMax activation')
        self.model.add(Dense(4, activation = 'softmax'))
        
        
        
    def load_model(self, model_loc):
        '''This function loads a model into the object. Input is a directory.'''
        self.reset_model()
        self.model = load_model(model_loc)
        
    def compile_model(self):
        self.model.compile(loss = "sparse_categorical_crossentropy", optimizer = self.optimizer, metrics=['accuracy'])
        
    def re_compile_model(self, opt = 0, lr = 0.0001):
        opt_list = [Nadam(learning_rate = lr), Adam(learning_rate = lr), SGD(learning_rate = lr)]
        print('Compiling')
        self.model.compile(loss = "sparse_categorical_crossentropy", optimizer = opt_list[opt], metrics=['accuracy'])
        print('Compiling complete')
        
    def display(self):
        try:
            self.model.summary()
        except:
            print('No Model Loaded')
            
    def grab_dataset(self, study):
        self.Image_train = load(f"E:\Brendan_2023_24\{study}\Image_train.sav", 'r')/244
        self.Image_test = load(f"E:\Brendan_2023_24\{study}\Image_test.sav", 'r')/244
        self.Label_train = load(f"E:\Brendan_2023_24\{study}\Label_train.sav", 'r')
        self.Label_test = load(f"E:\Brendan_2023_24\{study}\Label_test.sav", 'r')
        
    def clear_dataset(self):
        del self.Image_train, self.Image_test, self.Label_train, self.Label_test
        gc.collect()
        
    def train_model(self, epochs = 10, batch_size = 10, dataset = 'Dataset/Packed'):
        with device("/gpu:0"):
            startTime = datetime.now()
            print('Training')
            self.grab_dataset(dataset)
            
            NN = self.model.fit(self.Image_train,
                                self.Label_train,
                                batch_size = batch_size,
                                epochs = epochs,
                                validation_data = (self.Image_test, self.Label_test))
            
            save_dir = f'E:\Brendan_2023_24\CNN\{self.name}'
            self.model.save(f"E:\Brendan_2023_24\CNN\CNN_Saved_{self.name}.h5")
            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            # Training Prediction
            Training_Label_Predict = self.model.predict(self.Image_train)
            dump(Training_Label_Predict, save_dir + "\Training_Label_Predict.sav")
            
            # Testing Prediction
            Testing_Label_Predict = self.model.predict(self.Image_test)
            dump(Testing_Label_Predict, save_dir + "\Testing_Label_Predict.sav")
            
            dump(self.Label_train, save_dir + "\Label.train.sav")
            dump(self.Label_test, save_dir + "\Label_test.sav")
            
            self.clear_dataset()
            print('Training complete')
            print(datetime.now() - startTime)
            
    def reset_model(self):
        del self.model
        gc.collect()
        self.model = None
  
            
def main():
    Eyesight = EyesightManager(name='test_model2')
    Eyesight.build_eyesight()
    Eyesight.compile_model()
    Eyesight.display()
    Eyesight.train_model()
    Eyesight.reset_model()
        
if __name__ == "__main__":
    main()
        
        