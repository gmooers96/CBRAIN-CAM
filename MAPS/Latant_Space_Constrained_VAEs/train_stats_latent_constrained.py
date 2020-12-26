import math
import os 
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import argparse
import json 

import tensorflow as tf
import tensorflow_probability as tfp 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import netCDF4

import keras
from keras import layers
from keras import backend as K
from keras.models import Model
from keras.losses import binary_crossentropy, mse
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint

class AnnealingCallback(keras.callbacks.Callback):
    def __init__(self, epochs):
        super(AnnealingCallback, self).__init__()
        self.epochs = epochs 
        
    def on_epoch_begin(self, epoch, logs={}):
        new_kl_weight = epoch/self.epochs 
        K.set_value(self.model.kl_weight, new_kl_weight)
        print("Using updated KL Weight:", K.get_value(self.model.kl_weight))

class Sampling(keras.layers.Layer):
    def call(self, inputs):
        """
        TODO 
        """
        mean, log_var = inputs
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var/2) + mean

def kl_reconstruction_loss(latent_space_value, z_log_var, z_mean, z, vae, lambda_weight, latent_constraint_value):
    def _kl_reconstruction_loss(true, pred):
        """
        TODO 
        """
        
        print("z is",tf.shape(z))
        print("z is",z.get_shape)
        print("z is", K.int_shape(z))
        
        print("z_log_var is",tf.shape(z_log_var))
        print("z_log_var is",z_log_var.get_shape)
        print("z_log_var is", K.int_shape(z_log_var))
        
        print("z_mean is",tf.shape(z_mean))
        print("z_mean is",z_mean.get_shape)
        print("z_mean is", K.int_shape(z_mean))
        
        print("true is",tf.shape(true))
        print("true is",true.get_shape)
        print("true is", K.int_shape(true))
        
        true = tf.reshape(true, [-1, 128 * 30])

        x_mu = pred[:, :128*30]
        x_log_var = pred[:, 128*30:]
        # Gaussian reconstruction loss
        mse = -0.5 * K.sum(K.square(true - x_mu)/K.exp(x_log_var), axis=1)
        var_trace = -0.5 * K.sum(x_log_var, axis=1)
        log2pi = -0.5 * 128 * 30 * np.log(2 * np.pi)
        
        log_likelihood = mse + var_trace + log2pi
        #print("log likelihood shape", log_likelihood.shape)

        # NOTE: We don't take a mean here, since we first want to add the KL term
        reconstruction_loss = -log_likelihood

        # KL divergence loss
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=1)
        kl_loss *= -0.5
        
        
        print("true is",tf.shape(true))
        print("true is",true.get_shape)
        print("true is", K.int_shape(true))
        print("x_mu is",tf.shape(x_mu))
        print("x_mu is",x_mu.get_shape)
        print("x_mu is", K.int_shape(x_mu))
        
        
        print("true is",tf.shape(true))
        print("true is",true.get_shape)
        print("true is", K.int_shape(true))
        wprime_wprime = tf.math.reduce_mean(true, axis=1, keepdims=False, name=None)
        print("x_mu is",tf.shape(x_mu))
        print("x_mu is",x_mu.get_shape)
        print("x_mu is", K.int_shape(x_mu))
        
        print("reconstruction_loss is",tf.shape(reconstruction_loss))
        print("reconstruction_loss is",reconstruction_loss.get_shape)
        print("reconstruction_loss is", K.int_shape(reconstruction_loss))
        print("kl_loss is",tf.shape(kl_loss))
        print("kl_loss is",kl_loss.get_shape)
        print("kl_loss is", K.int_shape(kl_loss))
        
        #wsum = tf.math.reduce_mean(wprime_wprime, axis=0, keepdims=False, name=None)
        z_0 = z[:,0]
        print("z_0 is",tf.shape(z_0))
        print("z_0 is",z_0.get_shape)
        print("z_0 is", K.int_shape(z_0))
           
        theta_0 = tf.convert_to_tensor(1.0, dtype=None, dtype_hint=None, name=None)
        linear_loss = latent_constraint_value*(wprime_wprime-theta_0*z_0)
      
        print("linear_loss is",tf.shape(linear_loss))
        print("linear_loss is",linear_loss.get_shape)
        print("linear_loss is", K.int_shape(linear_loss))
       
        #print(gsdgsgs)
        #Frobenius_norm = K.sum(Frobenius_norm, axis = 1)
        #####################################################################################
        #return K.mean(reconstruction_loss + vae.kl_weight*kl_loss + lambda_weight*Frobenius_norm)
        #return K.mean(reconstruction_loss + vae.kl_weight*kl_loss+linear_loss)
        return K.mean(reconstruction_loss + vae.kl_weight*kl_loss)

    return _kl_reconstruction_loss

def kl(z_log_var, z_mean):
    def _kl(true, pred):
        """
        TODO 
        """
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return K.mean(kl_loss)
    
    return _kl

def reconstruction(true, pred):
    """
    TODO
    """
    true = tf.reshape(true, [-1, 128 * 30])

    x_mu = pred[:, :128*30]
    x_log_var = pred[:, 128*30:]

    mse = -0.5 * K.sum(K.square(true - x_mu)/K.exp(x_log_var), axis=1)
    var_trace = -0.5 * K.sum(x_log_var, axis=1)
    log2pi = -0.5 * 128 * 30 * np.log(2 * np.pi)
    
    log_likelihood = mse + var_trace + log2pi
    print("log likelihood shape", log_likelihood.shape)

    return K.mean(-log_likelihood)

def constrainer(z_log_var, z_mean, lambda_weight):
    def _constrainer(true, pred):
        true = tf.reshape(true, [-1, 128 * 30])
        x_mu = pred[:, :128*30]
        covariance_truth = tfp.stats.covariance(true)
        covariance_prediction = tfp.stats.covariance(x_mu)
        Frobenius_norm = tf.norm(covariance_prediction-covariance_truth, ord="euclidean")
        return lambda_weight*Frobenius_norm
        #return 1000000.0*Frobenius_norm
    return _constrainer

def linear_regression(latent_space_value, z_log_var, z_mean, z, latent_constraint_value):
    def _linear_regression(true, pred):
        true = tf.reshape(true, [-1, 128 * 30])
        z_0 = z[:,0]
        wprime_wprime = tf.math.reduce_mean(true, axis=1, keepdims=False, name=None)
        theta_0 = tf.convert_to_tensor(1.0, dtype=None, dtype_hint=None, name=None)
        linear_loss = latent_constraint_value*(wprime_wprime-theta_0*z_0)
        return K.mean(linear_loss)
    return _linear_regression

def encoder_gen(input_shape: tuple, encoder_config: dict, id):
    """
    Create the architecture for the VAE encoder. 
    """

    class EncoderResult():
        pass 

    encoder_result = EncoderResult()

    # Construct VAE Encoder layers
    inputs = keras.layers.Input(shape=[input_shape[0], input_shape[1], 1])
    zero_padded_inputs = keras.layers.ZeroPadding2D(padding=(1, 0))(inputs)

    print("shape of input after padding", inputs.shape)
    z = keras.layers.convolutional.Conv2D(
        encoder_config["conv_1"]["filter_num"], 
        tuple(encoder_config["conv_1"]["kernel_size"]), 
        padding='same', 
        activation=encoder_config["activation"], 
        strides=encoder_config["conv_1"]["stride"]
    )(zero_padded_inputs)
    print("shape after first convolutional layer", z.shape)

    z = keras.layers.convolutional.Conv2D(
        encoder_config["conv_2"]["filter_num"], 
        tuple(encoder_config["conv_2"]["kernel_size"]), 
        padding='same', 
        activation=encoder_config["activation"], 
        strides=encoder_config["conv_2"]["stride"]
    )(z)

    print("shape after second convolutional layer", z.shape)

    z = keras.layers.convolutional.Conv2D(
        encoder_config["conv_3"]["filter_num"], 
        tuple(encoder_config["conv_3"]["kernel_size"]), 
        padding='same', 
        activation=encoder_config["activation"], 
        strides=encoder_config["conv_3"]["stride"]
    )(z)

    print("shape after third convolutional layer", z.shape)

    z_mean = keras.layers.convolutional.Conv2D(
        encoder_config["conv_mu"]["filter_num"], 
        tuple(encoder_config["conv_mu"]["kernel_size"]), 
        padding='same', 
        strides=encoder_config["conv_mu"]["stride"]
    )(z)

    z_log_var = keras.layers.convolutional.Conv2D(
        encoder_config["conv_log_var"]["filter_num"], 
        tuple(encoder_config["conv_log_var"]["kernel_size"]), 
        padding='same', 
        strides=encoder_config["conv_log_var"]["stride"]
    )(z)

    z_mean = keras.layers.Flatten()(z_mean)
    z_log_var = keras.layers.Flatten()(z_log_var)

    print("z mean shape", z_mean._keras_shape)
    print("z log var shape", z_log_var._keras_shape)

    z = Sampling()([z_mean, z_log_var])

    # Instantiate Keras model for VAE encoder 
    vae_encoder = keras.Model(inputs=[inputs], outputs=[z_mean, z_log_var, z])
    plot_model(vae_encoder, to_file='./model_graphs/model_diagrams/encoder_{}.png'.format(id), show_shapes=True)
    # Package up everything for the encoder
    encoder_result.inputs = inputs
    encoder_result.z_mean = z_mean
    encoder_result.z_log_var = z_log_var
    encoder_result.z = z
    encoder_result.vae_encoder = vae_encoder 
    return encoder_result

def decoder_gen(
    original_input: tuple,
    decoder_config: dict
):
    """
    Create the architecture for the VAE decoder 
    """
    decoder_inputs = keras.layers.Input(shape=[decoder_config["latent_dim"]])
    print("decoder_inputs", decoder_inputs._keras_shape)
    # Reshape input to be an image 
    #for original case - for latent space size 64 - config 38
    #x = keras.layers.Reshape((2, 8, 4))(decoder_inputs)
    #superior for 1024 - works best - config 35
    x = keras.layers.Reshape((decoder_config["latent_reshape"]["dim_1"], decoder_config["latent_reshape"]["dim_2"], decoder_config["latent_reshape"]["dim_3"]))(decoder_inputs)
    #for foster arch - config 34
    #x = keras.layers.Reshape((2, 8, 32))(decoder_inputs)
    x = keras.layers.convolutional.Conv2DTranspose(
        decoder_config["conv_t_0"]["filter_num"], 
        tuple(decoder_config["conv_t_0"]["kernel_size"]), 
        padding='same', 
        activation=decoder_config["activation"], 
        strides=decoder_config["conv_t_0"]["stride"]
    )(x)

    # Start tranpose convolutional layers that upsample the image
    print("shape at beginning of decoder", x.shape)

    x = keras.layers.Conv2DTranspose(
        decoder_config["conv_t_1"]["filter_num"], 
        tuple(decoder_config["conv_t_1"]["kernel_size"]), 
        padding='same', 
        activation=decoder_config["activation"], 
        strides=decoder_config["conv_t_1"]["stride"]
    )(x)
    print("shape after first convolutional transpose layer", x._keras_shape)

    x = keras.layers.Conv2DTranspose(
        decoder_config["conv_t_2"]["filter_num"], 
        tuple(decoder_config["conv_t_2"]["kernel_size"]), 
        padding='same', 
        strides=decoder_config["conv_t_2"]["stride"],
        activation=decoder_config["activation"]
    )(x)
    print("shape after second convolutional transpose layer", x._keras_shape)
    x_mu = keras.layers.Conv2DTranspose(
        decoder_config["conv_mu"]["filter_num"], 
        tuple(decoder_config["conv_mu"]["kernel_size"]), 
        padding='same',  
        strides=decoder_config["conv_mu"]["stride"],
        activation=decoder_config["conv_mu"]["activation"]
    )(x)
    print("shape after convolutional mu layer", x_mu._keras_shape)

    x_log_var = keras.layers.Conv2DTranspose(
        decoder_config["conv_log_var"]["filter_num"], 
        tuple(decoder_config["conv_log_var"]["kernel_size"]), 
        padding='same',  
        strides=decoder_config["conv_log_var"]["stride"],
        activation=decoder_config["conv_log_var"]["activation"]
    )(x)
    print("shape after convolutional log var layer", x_mu._keras_shape)

    x_mu = keras.layers.Cropping2D(cropping=(1, 0))(x_mu)
    print("shape after cropping", x_mu._keras_shape)

    x_log_var = keras.layers.Cropping2D(cropping=(1, 0))(x_log_var)
    print("shape after cropping", x_log_var._keras_shape)

    x_mu = keras.layers.Flatten()(x_mu)
    x_log_var = keras.layers.Flatten()(x_log_var)
   
    x_mu_log_var = keras.layers.Concatenate(axis=1)([x_mu, x_log_var])
    
    variational_decoder = keras.Model(inputs=[decoder_inputs], outputs=[x_mu_log_var])
    
    return variational_decoder

def plot_training_losses(h, id):
    """
    Plot training loss graphs for 
        (1) KL term
        (2) Reconstruction term
        (3) Total ELBO loss  
    """
    hdict = h.history
    print(hdict)

    train_reconstruction_losses = hdict['reconstruction']
    valid_reconstruction_losses = hdict['val_reconstruction']

    kl_train_losses = hdict['_kl']
    kl_valid_losses = hdict['val__kl']
    
    constraint_train_losses = hdict['_linear_regression']
    constraint_valid_losses = hdict['val__linear_regression']

    total_train_losses = hdict['_kl_reconstruction_loss']
    total_valid_losses = hdict['val__kl_reconstruction_loss']

    epochs = range(1, len(train_reconstruction_losses) + 1)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12.8, 4.8))

    # Plot combined loss 
    ax1.plot(epochs, total_train_losses, 'b', label='Train')
    ax1.plot(epochs, total_valid_losses, 'r', label='Valid')
    ax1.set(xlabel="Epochs", ylabel="Loss")
    ax1.legend(prop={'size': 10})
    ax1.set_title("Combined Loss")

    # Plot KL 
    ax2.plot(epochs, kl_train_losses, 'b', label='Train')
    ax2.plot(epochs, kl_valid_losses, 'r', label='Valid')
    ax2.set(xlabel="Epochs", ylabel="Loss")
    ax2.legend(prop={'size': 10})
    ax2.set_title("KL Loss")

    # Plot reconstruction loss 
    ax3.plot(epochs, train_reconstruction_losses, 'b', label='Train')
    ax3.plot(epochs, valid_reconstruction_losses, 'r', label='Valid')
    ax3.set(xlabel="Epochs", ylabel="Loss")
    ax3.legend(prop={'size': 10})
    ax3.set_title("Reconstruction Loss")
    ax3.set_ylim(-25000, 10000)
    
    # Plot constraint loss 
    ax4.plot(epochs, constraint_train_losses, 'b', label='Train')
    ax4.plot(epochs, constraint_valid_losses, 'r', label='Valid')
    ax4.set(xlabel="Epochs", ylabel="Loss")
    ax4.legend(prop={'size': 10})
    ax4.set_title("Semisupervized Linear Term")
    
    plt.tight_layout()

    plt.savefig('./model_graphs/losses/model_losses_{}.png'.format(id))

def main():
    args = argument_parsing()
    print("Command line args:", args)

    f = open("./model_config/config_{}.json".format(args.id))
    model_config = json.load(f)
    f.close()

    train_data = np.load(model_config["data"]["training_data_path"])
    test_data = np.load(model_config["data"]["test_data_path"])

    img_width = train_data.shape[1]
    img_height = train_data.shape[2]

    print("Image shape:", img_width, img_height)
    
    # Construct VAE Encoder 
    encoder_result = encoder_gen((img_width, img_height), model_config["encoder"], args.id)
    # Construct VAE Decoder 
    vae_decoder = decoder_gen(
        (img_width, img_height),  
        model_config["decoder"]
    )
    plot_model(vae_decoder, to_file='./model_graphs/model_diagrams/decoder_{}.png'.format(args.id), show_shapes=True)
    _, _, z = encoder_result.vae_encoder(encoder_result.inputs)
    x_mu_log_var = vae_decoder(z)
    vae = keras.Model(inputs=[encoder_result.inputs], outputs=[x_mu_log_var])
    plot_model(vae, to_file='./model_graphs/model_diagrams/full_vae_{}.png'.format(args.id), show_shapes=True)
    vae.kl_weight = K.variable(model_config["kl_weight"])

    # Specify the optimizer 
    optimizer = keras.optimizers.Adam(lr=model_config['optimizer']['lr'])
    
    stat_weight = model_config['contraint_weight']['lambda']
    latent_weight = model_config['contraint_weight']['latent_constraint']
    
    latent_size = model_config['encoder']['latent_dim']
    # Compile model
    vae.compile(
        # loss=reconstruction, 
        loss=kl_reconstruction_loss(
            latent_size,
            encoder_result.z_log_var, 
            encoder_result.z_mean, 
            encoder_result.z,
            vae,
            stat_weight,
            latent_weight
        ), 
        optimizer=optimizer, 
        metrics=[
            reconstruction, 
            kl(
                encoder_result.z_log_var, 
                encoder_result.z_mean
            ), 
            kl_reconstruction_loss(
                latent_size,
                encoder_result.z_log_var, 
                encoder_result.z_mean,
                encoder_result.z,
                vae,
                stat_weight,
                latent_weight
            ), 
            linear_regression(
                latent_size,
                encoder_result.z_log_var, 
                encoder_result.z_mean,
                encoder_result.z,
                latent_weight
            )
        ]
    )
    
    vae.summary()

    train_data = train_data.reshape(train_data.shape+(1,))
    test_data = test_data.reshape(test_data.shape+(1,))

    print("train data shape", train_data.shape)
    print("test data shape", test_data.shape)


    checkpoint = ModelCheckpoint(
        './models/model_{}.th'.format(args.id), 
        monitor='val_loss', 
        verbose=1,
        save_best_only=True,
        save_weights_only=True 
    )
    callbacks_list = [checkpoint]

    if model_config["annealing"]:
        kl_weight_annealing = AnnealingCallback(model_config["train_epochs"])
        callbacks_list.append(kl_weight_annealing)

    h = vae.fit(
        x=train_data, 
        y=train_data, 
        epochs=model_config["train_epochs"], 
        batch_size=model_config["batch_size"], 
        validation_data=[test_data, test_data],
        callbacks=callbacks_list
    )

    plot_training_losses(h, args.id)

def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, help='This option specifies the config file to use to construct and train the VAE.')

    args = parser.parse_args()
    return args 

if __name__ == "__main__":
    main()