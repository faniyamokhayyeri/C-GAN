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
from helpers.refiner import build_refiner
from helpers.classifier import build_classifier
from helpers.discriminator import build_discriminator

class CGAN():
    def __init__(self):
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.dataset_name = 'chokepoint'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))

        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        self.gf = 32
        self.df = 64

        self.lambda_c = 10.0                    
        self.lambda_id = 0.1 * self.lambda_c    

        optimizer = Adam(0.0002, 0.5)

        self.D_R = build_discriminator(self.img_shape, self.df)
        self.D_F = build_discriminator(self.img_shape, self.df)
        self.D_R.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.D_F.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])


        self.Refiner = build_refiner(self.img_shape, self.gf, self.channels)
        self.Classifier = build_classifier(self.img_shape, self.gf, self.channels)

        img_sim = Input(shape=self.img_shape)
        img_target = Input(shape=self.img_shape)

        refineD_F = self.Refiner(img_sim)
        refineD_R = self.Classifier(img_target)

        rec_sim = self.Classifier(refineD_F)
        rec_target = self.Refiner(refineD_R)

        img_sim_id = self.Classifier(img_sim)
        img_target_id = self.Refiner(img_target)


        self.D_R.trainable = False
        self.D_F.trainable = False


        valiD_R = self.D_R(refineD_R)
        valiD_F = self.D_F(refineD_F)


        self.combined = Model(inputs=[img_sim, img_target],
                              outputs=[ valiD_R, valiD_F,
                                        rec_sim, rec_target,
                                        img_sim_id, img_target_id ])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                            loss_weights=[  1, 1,
                                            self.lambda_c, self.lambda_c,
                                            self.lambda_id, self.lambda_id ],
                            optimizer=optimizer)

    

    def train(self, epochs, batch_size=1, interval=50):

        start_time = datetime.datetime.now()

        valid = np.ones((batch_size,) + self.disc_patch)
        refined = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_sim, imgs_target) in enumerate(self.data_loader.load_batch(batch_size)):

                refineD_F = self.Refiner.predict(imgs_sim)
                refineD_R = self.Classifier.predict(imgs_target)

                dsim_loss_real = self.D_R.train_on_batch(imgs_sim, valid)
                dsim_loss_refined = self.D_R.train_on_batch(refineD_R, refined)
                dsim_loss = 0.5 * np.add(dsim_loss_real, dsim_loss_refined)

                dtarget_loss_real = self.D_F.train_on_batch(imgs_target, valid)
                dtarget_loss_refined = self.D_F.train_on_batch(refineD_F, refined)
                dtarget_loss = 0.5 * np.add(dtarget_loss_real, dtarget_loss_refined)

                d_loss = 0.5 * np.add(dsim_loss, dtarget_loss)

                g_loss = self.combined.train_on_batch([imgs_sim, imgs_target],
                                                        [valid, valid,
                                                        imgs_sim, imgs_target,
                                                        imgs_sim, imgs_target])

                elapsed_time = datetime.datetime.now() - start_time

                print ("[Epoch %d/%d] [targetatch %d/%d] [DR loss: %f, acc: %3d%%] [R loss: %05f, adv: %05f, DF: %05f, id: %05f] time: %s " \
                                                                        % ( epoch, epochs,
                                                                            batch_i, self.data_loader.n_batches,
                                                                            d_loss[0], 100*d_loss[1],
                                                                            g_loss[0],
                                                                            np.mean(g_loss[1:3]),
                                                                            np.mean(g_loss[3:5]),
                                                                            np.mean(g_loss[5:6]),
                                                                            elapsed_time))

                if batch_i % interval == 0:
                    self.sample_images(epoch, batch_i)

    def sample_images(self, epoch, batch_i):
        os.makedirs('output/%s' % self.dataset_name, exist_ok=True)
        r, c = 1, 3

        imgs_sim = self.data_loader.load_data(domain="sim", batch_size=1, is_testing=True)
        imgs_target = self.data_loader.load_data(domain="target", batch_size=1, is_testing=True)

        refineD_F = self.Refiner.predict(imgs_sim)
        refineD_R = self.Classifier.predict(imgs_target)

        rec_sim = self.Classifier.predict(refineD_F)
        rec_target = self.Refiner.predict(refineD_R)

        gen_imgs = np.concatenate([imgs_sim, refineD_F, rec_sim, imgs_target, refineD_R, rec_target])

        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Simulated', 'Refined','Target']
        fig, axs = plt.subplots(r, c)

        axs[0].imshow(gen_imgs[0])
        axs[0].set_title(titles[0])
        axs[0].axis('off')

        axs[1].imshow(gen_imgs[1])
        axs[1].set_title(titles[1])
        axs[1].axis('off')

        axs[2].imshow(gen_imgs[3])
        axs[2].set_title(titles[2])
        axs[2].axis('off')

        fig.savefig("output/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        plt.close()


if __name__ == '__main__':
    cgan = CGAN()
    cgan.train(epochs=100, batch_size=1, interval=50)
