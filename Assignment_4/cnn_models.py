from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.merge import add
from keras import regularizers
from keras import initializers
from keras import constraints
from keras.engine import Layer
from keras.engine import InputSpec
from keras.utils import conv_utils
from keras.legacy import interfaces
#import depth_class
from keras.layers import DepthwiseConv2D
#import DepthwiseConv2D
#from keras.applications import mobilenet
from keras import backend as K

def resnet8(img_width, img_height, img_channels, output_dim):
    """
    Define model architecture.
    
    # Arguments
       img_width: Target image widht.
       img_height: Target image height.
       img_channels: Target image channels.
       output_dim: Dimension of model output.
       
    # Returns
       model: A Model instance.
    """

    # Input
    img_input = Input(shape=(img_height, img_width, img_channels))

    x1 = Conv2D(32, (5, 5), strides=[2,2], padding='same')(img_input)
    x1 = MaxPooling2D(pool_size=(3, 3), strides=[2,2])(x1)

    # First residual block
    x2 = keras.layers.normalization.BatchNormalization()(x1)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x2)

    x2 = keras.layers.normalization.BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x2)

    x1 = Conv2D(32, (1, 1), strides=[2,2], padding='same')(x1)
    x3 = add([x1, x2])

    # Second residual block
    x4 = keras.layers.normalization.BatchNormalization()(x3)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x4)

    x4 = keras.layers.normalization.BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x4)

    x3 = Conv2D(64, (1, 1), strides=[2,2], padding='same')(x3)
    x5 = add([x3, x4])

    # Third residual block
    x6 = keras.layers.normalization.BatchNormalization()(x5)
    x6 = Activation('relu')(x6)
    x6 = Conv2D(128, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x6)

    x6 = keras.layers.normalization.BatchNormalization()(x6)
    x6 = Activation('relu')(x6)
    x6 = Conv2D(128, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x6)

    x5 = Conv2D(128, (1, 1), strides=[2,2], padding='same')(x5)
    x7 = add([x5, x6])

    x = Flatten()(x7)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    # Steering channel
    steer = Dense(output_dim)(x)

    # Collision channel
    coll = Dense(output_dim)(x)
    coll = Activation('sigmoid')(coll)

    # Define steering-collision model
    model = Model(inputs=[img_input], outputs=[steer, coll])
    print(model.summary())

    return model
'''
def MobileNet(input_shape=None,
              alpha=1.0,
              depth_multiplier=1,
              dropout=1e-3,
              include_top=True,
              weights='imagenet',
              input_tensor=None,
              pooling=None,
              classes=1000,
              output_dim):

'''

def MobileNet(img_width, img_height, img_channels, output_dim,
              depth_multiplier=1,input_tensor=None,
              dropout=1e-3):
    """
    Define model architecture.
    
    # Arguments
       img_width: Target image widht.
       img_height: Target image height.
       img_channels: Target image channels.
       output_dim: Dimension of model output.
       
    # Returns
       model: A Model instance.
    """

    # Input
    img_input = Input(shape=(img_height, img_width, img_channels))


    x1 = Conv2D(16, (5, 5), strides=[2,2], padding='same')(img_input)
    x1 = MaxPooling2D(pool_size=(3, 3), strides=[2,2])(x1)

    # First residual block
    x2=DepthwiseConv2D((3, 3),
                        padding='valid',
                        depth_multiplier=depth_multiplier,
                        strides=(2,2),
                        use_bias=False)(x1)
    x2 = keras.layers.normalization.BatchNormalization()(x2)
    x2 = Activation('relu')(x2)

    x2 = Conv2D(32,(1, 1), padding='same', use_bias=False, strides=(1, 1))(x2)
    x2 = keras.layers.normalization.BatchNormalization()(x2)
    x2 = Activation('relu')(x2)

    x2 = MaxPooling2D(pool_size=(3, 3), strides=[2,2])(x2)

    x2 = Conv2D(32, (5, 5), strides=[2,2], padding='same')(x2)
    x2 = MaxPooling2D(pool_size=(3, 3), strides=[2,2])(x2)
    
    x2=DepthwiseConv2D((2, 2),
                         padding='valid',
                         depth_multiplier=depth_multiplier,
                         strides=(1,1),
                         use_bias=False)(x2)


    x2 = keras.layers.normalization.BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(64,(1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1))(x2)

    x2 = keras.layers.normalization.BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    
    x3 = Flatten()(x2)
    x3 = Activation('relu')(x3)
    x3 = Dropout(0.5)(x3)

    # Steering channel
    steer = Dense(output_dim)(x3)

    # Collision channel
    coll = Dense(output_dim)(x3)
    coll = Activation('sigmoid')(coll)

    # Define steering-collision model
    model = Model(inputs=[img_input], outputs=[steer, coll])
    print(model.summary())

    return model

