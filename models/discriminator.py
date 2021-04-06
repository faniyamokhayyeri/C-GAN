from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers import  MaxPooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

def build_discriminator(img_shape):
    img = Input(shape=img_shape)

    d1 = Conv2D(32, 3, 3, border_mode='same', activation='relu')(img)
    d1 = MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='same')(d1)

    d2 = Conv2D(64, 3, 3, border_mode='same', activation='relu')(d1)
    d2 = MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='same')(d2)

    d3 = Conv2D(128, 3, 3, border_mode='same', activation='relu')(d2)
    d3 = MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='same')(d3)

    d5 = Flatten()(d3)
    d6 = Dense(1, activation='sigmoid')(d5)

    return Model(img, d6)

def build_feature_discriminator(n_features):
    f = Input(shape=(n_features,))
    d = Dense(1, activation='sigmoid')(f)

    return Model(f, d)