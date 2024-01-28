#------------------------ IMPORTS AND CONSTANTS ------------------------
# Import required libraries
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.utils import to_categorical
from keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam

# Define constants
IMG_HEIGHT = 256 #1280
IMG_WIDTH = 320 #1024
NUM_CLASSES = 2
LATENT_DIM = 100 #1000
BATCH_SIZE = 32
EPOCHS = 10 #100


#------------------------ PREPROCESSING ------------------------
# Load images into memory
image_list = []
label_list = []
for filename in os.listdir('dataset'):
    img = Image.open(os.path.join('dataset', filename))
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img = np.array(img)
    image_list.append(img)
    if 'alive' in filename:
        label_list.append(0)
    else:
        label_list.append(1)
images = np.array(image_list)
labels = to_categorical(label_list, NUM_CLASSES)

# Preprocess images
images = images.astype('float32') / 255.0


#------------------------ FUNCTIONS FOR THE GENERATIVE ADVERSERIAL NETWORK MODEL ------------------------
def build_generator():
    # Define generator architecture
    input_layer = Input(shape=(LATENT_DIM,))
    # Calculate size of final flattened layer in generator
    gen_flat_size = (IMG_HEIGHT // (2 ** 3)) * (IMG_WIDTH // (2 ** 3)) * 128
    #x = Dense(8 * 8 * 256)(input_layer)
    x = Dense(gen_flat_size)(input_layer)
    x = LeakyReLU(alpha=0.2)(x)
    #x = Reshape((8, 8, 256))(x)
    x = Reshape((IMG_HEIGHT // (2 ** 3), IMG_WIDTH // (2 ** 3), 128))(x)
    x = Conv2DTranspose(128, kernel_size=5, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(64, kernel_size=5, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(32, kernel_size=5, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    output_layer = Conv2D(3, kernel_size=5, activation='sigmoid', padding='same')(x)
    generator = Model(input_layer, output_layer)
    return generator


def build_discriminator():
    # Define discriminator architecture
    input_layer = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = Conv2D(32, kernel_size=5, strides=2, padding='same')(input_layer)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(64, kernel_size=5, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(128, kernel_size=5, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    """ x = Conv2D(256, kernel_size=5, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(512, kernel_size=5, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x) """
    x = Flatten()(x)
    output_layer = Dense(1, activation='sigmoid')(x)
    discriminator = Model(input_layer, output_layer)
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])
    return discriminator


def build_gan(generator, discriminator):
    # Define GAN architecture
    discriminator.trainable = False #Freeze discriminator layers
    input_layer = Input(shape=(LATENT_DIM,))
    x = generator(input_layer)
    output_layer = discriminator(x)
    gan = Model(input_layer, output_layer)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
    return gan


def train(images, labels):
    # Split data into training and testing sets
    train_images = images[:800]
    train_labels = labels[:800]
    test_images = images[800:]
    test_labels = labels[800:]

    # Initialize models
    generator = build_generator()
    discriminator = build_discriminator()
    gan = build_gan(generator, discriminator)

    # Train models
    for epoch in range(EPOCHS):
        for batch in range(int(train_images.shape[0] / BATCH_SIZE)):
            # Train discriminator
            noise = np.random.normal(0, 1, size=(BATCH_SIZE, LATENT_DIM))
            generated_images = generator.predict(noise)
            real_images = train_images[batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE]
            x = np.concatenate((real_images, generated_images))
            y = np.zeros((2 * BATCH_SIZE, 1))  # y = np.zeros((2 * BATCH_SIZE, 1)) 
            y[:BATCH_SIZE] = 0.9  # One-sided label smoothing for real images
            y[BATCH_SIZE:] = 0.1  # One-sided label smoothing for fake images
            discriminator_loss = discriminator.train_on_batch(x, y)

            # Train generator
            noise = np.random.normal(0, 1, size=(BATCH_SIZE, LATENT_DIM))
            y = np.ones((BATCH_SIZE))  # y = np.ones((BATCH_SIZE, 1))
            generator_loss = gan.train_on_batch(noise, y)

        # Evaluate models
        accuracy = discriminator.evaluate(test_images, np.ones((test_images.shape[0], 1)), verbose=0)[1]  # accuracy = discriminator.evaluate(test_images, test_labels, verbose=0)[1]

        # Print progress
        print(f'Epoch: {epoch + 1}/{EPOCHS}, Discriminator Loss: {discriminator_loss[0]}, '
              f'Discriminator Accuracy: {accuracy}, Generator Loss: {generator_loss}')
        
    # Save models
    generator.save('generator.h5')
    discriminator.save('discriminator.h5')
    gan.save('gan.h5')
    
        
def generate_images(generator):
    # Generate new images
    noise = np.random.normal(0, 1, size=(25, LATENT_DIM))
    generated_images = generator.predict(noise)

    # Save images to disk
    for i in range(generated_images.shape[0]):
        img = generated_images[i]
        img = np.uint8(img * 255.0)
        img = Image.fromarray(img)
        img.save(f'generated_image_{i}.png')


#------------------------ MAIN ------------------------
def main():
    """ generator = build_generator()
    discriminator = build_discriminator()
    build_gan(generator, discriminator) """
    train(images, labels)  # comment out this line for using the already trained model
    
    # Load generator
    generator = load_model('generator.h5')

    # Load discriminator
    discriminator = load_model('discriminator.h5')

    # Load GAN
    gan = load_model('gan.h5')
    
    # Compile the generator model
    generator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')

    # Compile the discriminator model
    discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])

    # Set the discriminator to non-trainable
    discriminator.trainable = False

    # Combine the generator and discriminator models to create the GAN
    gan_input = tf.keras.layers.Input(shape=(LATENT_DIM,))
    gan_output = discriminator(generator(gan_input))
    gan = tf.keras.models.Model(gan_input, gan_output)

    # Compile the GAN model
    gan.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')
    
    """ num_generated_images=5
    for i in range(num_generated_images):
        generate_images(generator) """
    
    # Generate images   
    generate_images(generator)
    
    """ # set hyperparameters
    latent_dim = [100, 1000, 2000, 5000]
    n_epochs = [100, 1000, 100000, 1000000]
    n_batch = [64, 128, 256, 512 ]
    
    # load and preprocess data
    dataset = load_data()
    # train GAN model
    gan_model = define_gan(generator, discriminator)
    train(gan_model, dataset, latent_dim, n_epochs, n_batch)
    # generate images
    generate_images(generator, latent_dim, num_images=10) """

if __name__ == '__main__':
    main()