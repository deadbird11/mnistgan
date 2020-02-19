import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import os

noise_dim = 100
BUFFER_SIZE = 60000
BATCH_SIZE = 64
EPOCHS = 50
display_interval = 2

(train_images, train_labels), (_, _) = mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

def make_generator_model():
    model = tf.keras.Sequential()

    model.add(layers.Dense(7*7*256, input_dim=noise_dim))
    model.add(layers.Reshape((7, 7, 256)))

    # CONV 1
    model.add(layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
    # maybe add axis=3?
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    # CONV 2
    model.add(layers.Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    # CONV 3
    model.add(layers.Conv2DTranspose(1, kernel_size=2, strides=2))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('tanh'))

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()

    # CONV 1
    model.add(layers.Conv2D(64, kernel_size=2, strides=2, padding='same', input_shape=(28,28,1)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    # CONV 2
    model.add(layers.Conv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    # CONV 3
    model.add(layers.Conv2D(256, kernel_size=3, strides=1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Reshape((7*7*256,)))
    model.add(layers.Dense(1))
    model.add(layers.LeakyReLU(alpha=0.2))

    return model

def discriminator_loss(real_pred, fake_pred):
    return tf.reduce_mean(real_pred) - tf.reduce_mean(fake_pred)

def generator_loss(fake_pred):
    return -tf.reduce_mean(fake_pred)

generator = make_generator_model()
discriminator = make_discriminator_model()

gen_optimizer = tf.keras.optimizers.Adam(1e-4)
disc_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=gen_optimizer,
                                 discriminator_optimizer=disc_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_images = generator(noise, training=True)

        real_pred = discriminator(images, training=True)
        fake_pred = discriminator(gen_images, training=True)

        gen_loss = generator_loss(fake_pred)
        disc_loss = discriminator_loss(real_pred, fake_pred)

    gen_grad = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_grad = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(gen_grad, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_grad, discriminator.trainable_variables))

def train(dataset, epochs):
    seed = tf.random.normal([BATCH_SIZE, noise_dim])

    for epoch in range(epochs):
        for batch in dataset:
            train_step(batch)

        print("On epoch ", epoch)

        if (epoch + 1) % display_interval == 0:
            generate_and_save_images(generator, epoch, seed)

        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)


def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i + 1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

train(train_dataset, EPOCHS)