import os
import math
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from model import get_generator
from model import get_encoder
from data import get_celebA
from config import flags


def eval(weights_path, image_path='samples.png', samples=64, x_shape=[None, 64, 64, 3], z_shape=[None, 128]):
    num_tiles = int(math.ceil(math.sqrt(samples)))
    G = get_generator(x_shape, z_shape)
    G.load_weights(weights_path, format='npz')
    G.eval()
    z_shape[0] = samples
    z = np.random.normal(loc=0.0, scale=1.0, size=z_shape).astype(np.float32)
    tl.visualize.save_images(G(z).numpy(), [num_tiles, num_tiles], image_path)
    
def evalencoder(weights_path, image_path='samples.png', samples=64, x_shape=[None, 64, 64, 3], z_shape=[None, 128]):
    z_real=[]
    num_tiles = int(math.ceil(math.sqrt(samples)))
    images, images_path = get_celebA(flags.output_size, flags.batch_size)
    E=get_encoder(x_shape, z_shape)
    E.load_weights(weights_path, format='npz')
    E.eval()
    for step, batch_images in enumerate(images):
        x_real = batch_images
        z_real.append(E(x_real).numpy())
        #tl.visualize.save_images(E(x_real).numpy(), image_path)
        print(z_real)

if __name__ == '__main__':
    evalencoder('drive/My\ Drive/bigancheckpints/10-E.npz')
