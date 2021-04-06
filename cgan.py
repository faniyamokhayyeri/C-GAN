import scipy
import datetime
import matplotlib.pyplot as plt
import sys
from loader import DataLoader
import numpy as np
import os
from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from models.refiner import build_refiner
from models.classifier import build_classifier
from models.discriminator import build_discriminator, build_feature_discriminator
from models.encoder import build_encoder

class CGAN():
    def __init__(self):
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.n_features = 128
        self.n_classes = 31
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.data_loader = DataLoader(img_res=(self.img_rows, self.img_cols), n_classes=self.n_classes)

        optimizer = Adam(0.0002, 0.5)

        self.D_R = build_discriminator(self.img_shape)
        self.D_F = build_feature_discriminator(self.n_features)
        self.D_R.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.D_F.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])


        self.Refiner = build_refiner(self.img_shape, self.channels)
        self.Feature = build_encoder(self.img_shape, self.n_features)
        self.Classifier = build_classifier(self.n_features, self.n_classes)

        self.D_R.trainable = False
        self.D_F.trainable = False

        self.Classifier.compile(loss='categorical_crossentropy',
                                optimizer=optimizer,
                                metrics=['accuracy'])
        self.Classifier.trainable = False

        self.GAN_1 = Sequential()
        self.GAN_1.add(self.Refiner)
        self.GAN_1.add(self.D_R)
        self.GAN_1.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        self.GAN_2 = Sequential()
        self.GAN_2.add(self.Refiner)
        self.GAN_2.add(self.Feature)
        self.GAN_2.add(self.D_F)
        self.GAN_2.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        self.GAN_3 = Sequential()
        self.GAN_3.add(self.Refiner)
        self.GAN_3.add(self.Feature)
        self.GAN_3.add(self.Classifier)
        self.GAN_3.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def train(self, epochs, batch_size=1, interval=50):

        start_time = datetime.datetime.now()

        valid = np.ones((batch_size,))
        refined = np.zeros((batch_size,))

        for epoch in range(epochs):
            for batch_i, (imgs_sim, imgs_target, classes) in enumerate(self.data_loader.load_batch(batch_size)):

                imgs_refined = self.Refiner.predict(imgs_sim)
                feature_sim = self.Feature.predict(imgs_sim)
                feature_target = self.Feature.predict(imgs_target)
                feature_refined = self.Feature.predict(imgs_refined)
                
                dimg_loss_real = self.D_R.train_on_batch(imgs_target, valid)
                dimg_loss_refined = self.D_R.train_on_batch(imgs_refined, refined)
                dimg_loss = 0.5 * np.add(dimg_loss_real, dimg_loss_refined)

                dfeature_loss_real = self.D_F.train_on_batch(feature_target, valid)
                dfeature_loss_refined = self.D_F.train_on_batch(feature_refined, refined)
                dfeature_loss = 0.5 * np.add(dfeature_loss_real, dfeature_loss_refined)
                
                class_loss = self.Classifier.train_on_batch(feature_sim, classes)

                gan1_loss = self.GAN_1.train_on_batch(imgs_sim, valid)
                gan2_loss = self.GAN_2.train_on_batch(imgs_sim, valid)
                gan3_loss = self.GAN_3.train_on_batch(imgs_sim, classes)

                elapsed_time = datetime.datetime.now() - start_time

                print ("[Epoch %d/%d] [targetatch %d/%d] [DR loss: %f] [DF loss: %f] [C loss: %f] [GAN_1 loss  %f] [GAN_2 loss  %f] [GAN_3 loss  %f] time: %s " \
                                                                        % ( epoch, epochs,
                                                                            batch_i, self.data_loader.n_batches,
                                                                            dimg_loss[0],
                                                                            dfeature_loss[0],
                                                                            class_loss[0],
                                                                            gan1_loss[0],
                                                                            gan2_loss[0],
                                                                            gan3_loss[0],
                                                                            elapsed_time))

                if batch_i % interval == 0:
                    self.sample_images(epoch, batch_i)

    def sample_images(self, epoch, batch_i):
        os.makedirs('output/', exist_ok=True)
        r, c = 1, 3

        imgs_sim = self.data_loader.load_data(domain="sim", batch_size=1, is_testing=True)
        imgs_target = self.data_loader.load_data(domain="target", batch_size=1, is_testing=True)
        imgs_refined = self.Refiner.predict(imgs_sim)

        gen_imgs = np.concatenate([imgs_sim, imgs_refined, imgs_target])

        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Simulated', 'Refined','Target']
        fig, axs = plt.subplots(r, c)

        axs[0].imshow(gen_imgs[0])
        axs[0].set_title(titles[0])
        axs[0].axis('off')

        axs[1].imshow(gen_imgs[1])
        axs[1].set_title(titles[1])
        axs[1].axis('off')

        axs[2].imshow(gen_imgs[2])
        axs[2].set_title(titles[2])
        axs[2].axis('off')

        fig.savefig("output/%d_%d.png" % (epoch, batch_i))
        plt.close()


if __name__ == '__main__':
    cgan = CGAN()
    cgan.train(epochs=100, batch_size=8, interval=50)
