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

        self.d_sim = self.build_discriminator()
        self.d_target = self.build_discriminator()
        self.d_sim.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.d_target.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])


        self.g_R1 = self.build_refiner()
        self.g_R2 = self.build_refiner()

        img_sim = Input(shape=self.img_shape)
        img_target = Input(shape=self.img_shape)

        refined_target = self.g_R1(img_sim)
        refined_sim = self.g_R2(img_target)

        rec_sim = self.g_R2(refined_target)
        rec_target = self.g_R1(refined_sim)

        img_sim_id = self.g_R2(img_sim)
        img_target_id = self.g_R1(img_target)


        self.d_sim.trainable = False
        self.d_target.trainable = False


        valid_sim = self.d_sim(refined_sim)
        valid_target = self.d_target(refined_target)


        self.combined = Model(inputs=[img_sim, img_target],
                              outputs=[ valid_sim, valid_target,
                                        rec_sim, rec_target,
                                        img_sim_id, img_target_id ])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                            loss_weights=[  1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id ],
                            optimizer=optimizer)

    def build_refiner(self):

        def conv2d(layer_input, filters, f_size=4):
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = InstanceNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u


        d0 = Input(shape=self.img_shape)


        d1 = conv2d(d0, self.gf)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)


        u1 = deconv2d(d4, d3, self.gf*4)
        u2 = deconv2d(u1, d2, self.gf*2)
        u3 = deconv2d(u2, d1, self.gf)

        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = Input(shape=self.img_shape)

        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(img, validity)

    def train(self, epochs, batch_size=1, interval=50):

        start_time = datetime.datetime.now()

        valid = np.ones((batch_size,) + self.disc_patch)
        refined = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_sim, imgs_target) in enumerate(self.data_loader.load_batch(batch_size)):

                refined_target = self.g_R1.predict(imgs_sim)
                refined_sim = self.g_R2.predict(imgs_target)

                dsim_loss_real = self.d_sim.train_on_batch(imgs_sim, valid)
                dsim_loss_refined = self.d_sim.train_on_batch(refined_sim, refined)
                dsim_loss = 0.5 * np.add(dsim_loss_real, dsim_loss_refined)

                dtarget_loss_real = self.d_target.train_on_batch(imgs_target, valid)
                dtarget_loss_refined = self.d_target.train_on_batch(refined_target, refined)
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

                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)

    def sample_images(self, epoch, batch_i):
        os.makedirs('output/%s' % self.dataset_name, exist_ok=True)
        r, c = 2, 3

        imgs_sim = self.data_loader.load_data(domain="sim", batch_size=1, is_testing=True)
        imgs_target = self.data_loader.load_data(domain="target", batch_size=1, is_testing=True)

        refined_target = self.g_R1.predict(imgs_sim)
        refined_sim = self.g_R2.predict(imgs_target)

        rec_sim = self.g_R2.predict(refined_target)
        rec_target = self.g_R1.predict(refined_sim)

        gen_imgs = np.concatenate([imgs_sim, refined_target, rec_sim, imgs_target, refined_sim, rec_target])

        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Simulated', 'Refined']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[j])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        plt.close()


if __name__ == '__main__':
    cgan = CGAN()
    cgan.train(epochs=100, batch_size=1, interval=50)
