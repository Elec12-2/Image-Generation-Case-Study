import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, LeakyReLU, UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import os

def build_generator():
    model = Sequential()
    model.add(Dense(128 * 7 * 7, activation="relu", input_dim=100))
    model.add(Reshape((7, 7, 128)))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(1, kernel_size=3, padding="same", activation='tanh'))
    model.summary()
    noise = Input(shape=(100,))
    img = model(noise)
    return Model(noise, img)

def build_discriminator():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=(28, 28, 1), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    img = Input(shape=(28, 28, 1))
    validity = model(img)
    return Model(img, validity)

def build_gan(generator, discriminator):
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
    discriminator.trainable = False
    noise = Input(shape=(100,))
    img = generator(noise)
    validity = discriminator(img)
    gan = Model(noise, validity)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return gan

def train_gan(generator, discriminator, gan, epochs, batch_size=128, save_interval=50):
    (X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]
        noise = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(noise)
        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        noise = np.random.normal(0, 1, (batch_size, 100))
        g_loss = gan.train_on_batch(noise, valid)

        print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {d_loss[1]*100}] [G loss: {g_loss}]")

        if epoch % save_interval == 0:
            save_imgs(generator, epoch)

def save_imgs(generator, epoch, img_dir="generated_images"):
    os.makedirs(img_dir, exist_ok=True)
    noise = np.random.normal(0, 1, (25, 100))
    gen_imgs = generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5
    fig, axs = plt.subplots(5, 5)
    cnt = 0
    for i in range(5):
        for j in range(5):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(f"{img_dir}/mnist_{epoch}.png")
    plt.close()

if __name__ == "__main__":
    generator = build_generator()
    discriminator = build_discriminator()
    gan = build_gan(generator, discriminator)
    train_gan(generator, discriminator, gan, epochs=10000, batch_size=64, save_interval=200)

import cv2
from matplotlib import pyplot as plt

def load_images(user_image_path, product_image_path):
    user_img = cv2.imread(user_image_path)
    product_img = cv2.imread(product_image_path, cv2.IMREAD_UNCHANGED)
    return user_img, product_img

def blend_images(user_img, product_img, position):
    x_offset, y_offset = position
    y1, y2 = y_offset, y_offset + product_img.shape[0]
    x1, x2 = x_offset, x_offset + product_img.shape[1]
    alpha_s = product_img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(0, 3):
        user_img[y1:y2, x1:x2, c] = (alpha_s * product_img[:, :, c] + alpha_l * user_img[y1:y2, x1:x2, c])
    return user_img

def display_image(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    user_img_path = 'path_to_user_image.jpg'
    product_img_path = 'path_to_product_image.png'
    user_img, product_img = load_images(user_img_path, product_img_path)
    position = (100, 150)  # Example position
    blended_img = blend_images(user_img, product_img, position)
    display_image(blended_img)