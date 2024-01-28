from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

LATENT_DIM = 100 


def generate_images(generator):
    # Generate new images
    noise = np.random.normal(0, 1, size=(25, LATENT_DIM))
    generated_images = generator.predict(noise)

    # Save images to disk
    for i in range(generated_images.shape[0]-1):
        img = generated_images[i]
        img = np.uint8(img * 255.0)
        img = Image.fromarray(img)
        img.save(f'generated_image_{i}.png')


# Load generator
generator = load_model('generator.h5')

# Generate image
generate_images(generator)
        

