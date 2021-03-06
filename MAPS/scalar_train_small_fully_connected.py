import math
import os 
os.environ["CUDA_VISIBLE_DEVICES"]="0"
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

def kl_reconstruction_loss(z_log_var, z_mean, vae, lambda_weight):
    def _kl_reconstruction_loss(true, pred):
        """
        TODO 
        """
        true = tf.reshape(true, [-1, 128])

        x_mu = pred[:, :128]
        x_log_var = pred[:, 128:]
        # Gaussian reconstruction loss
        mse = -0.5 * K.sum(K.square(true - x_mu)/K.exp(x_log_var), axis=1)
        var_trace = -0.5 * K.sum(x_log_var, axis=1)
        log2pi = -0.5 * 128 * np.log(2 * np.pi)
        
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
        #print(fgdfdfgdfag)
        
        covariance_truth = tfp.stats.covariance(true)
        covariance_prediction = tfp.stats.covariance(x_mu)
        Frobenius_norm = tf.norm(covariance_prediction-covariance_truth, ord="euclidean")
        
        
        print("true is",tf.shape(true))
        print("true is",true.get_shape)
        print("true is", K.int_shape(true))
        print("x_mu is",tf.shape(x_mu))
        print("x_mu is",x_mu.get_shape)
        print("x_mu is", K.int_shape(x_mu))

        #Frobenius_norm = K.sum(Frobenius_norm, axis = 1)
  
        #print("Frobenius_norm is",tf.shape(Frobenius_norm))
        #print("Frobenius_norm is",Frobenius_norm.get_shape)
        
        print("reconstruction_loss is",tf.shape(reconstruction_loss))
        print("reconstruction_loss is",reconstruction_loss.get_shape)
        print("reconstruction_loss is", K.int_shape(reconstruction_loss))
        print("kl_loss is",tf.shape(kl_loss))
        print("kl_loss is",kl_loss.get_shape)
        print("kl_loss is", K.int_shape(kl_loss))
        #print(gsdgsgs)
        #Frobenius_norm = K.sum(Frobenius_norm, axis = 1)
        #####################################################################################
        #return K.mean(reconstruction_loss + vae.kl_weight*kl_loss + lambda_weight*Frobenius_norm)
        return K.mean(reconstruction_loss + vae.kl_weight*kl_loss)# + lambda_weight*Frobenius_norm)
        

    return _kl_reconstruction_loss

def kl(z_log_var, z_mean):
    def _kl(true, pred):
        """
        TODO 
        """
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        # kl_loss = K.print_tensor(kl_loss, message='EULA PEULA')
        return K.mean(kl_loss)
    
    return _kl

def reconstruction(true, pred):
    """
    TODO
    """
    true = tf.reshape(true, [-1, 128])

    x_mu = pred[:, :128]
    x_log_var = pred[:, 128:]

    mse = -0.5 * K.sum(K.square(true - x_mu)/K.exp(x_log_var), axis=1)
    var_trace = -0.5 * K.sum(x_log_var, axis=1)
    log2pi = -0.5 * 128 * np.log(2 * np.pi)
    
    log_likelihood = mse + var_trace + log2pi
    print("log likelihood shape", log_likelihood.shape)

    return K.mean(-log_likelihood)

def constrainer(z_log_var, z_mean, lambda_weight):
    def _constrainer(true, pred):
        true = tf.reshape(true, [-1, 128])
        x_mu = pred[:, :128]
        covariance_truth = tfp.stats.covariance(true)
        covariance_prediction = tfp.stats.covariance(x_mu)
        Frobenius_norm = tf.norm(covariance_prediction-covariance_truth, ord="euclidean")
        return lambda_weight*Frobenius_norm
        #return 1000000.0*Frobenius_norm
    return _constrainer

def power_spectrum(z_log_var, z_mean):
    def _power_spectrum(true, pred):
        p850 = tf.reshape(pred[22,:], [-1, 128 ])
        t850 = tf.reshape(true[22,:], [-1, 128 ])
        p850 = tf.cast(p850, dtype=tf.float32)
        t850 = tf.cast(t850, dtype=tf.float32)
        P_pred = tf.signal.rfft(p850)*tf.math.conj(tf.signal.rfft(p850))
        P_truth = tf.signal.rfft(t850)*tf.math.conj(tf.signal.rfft(t850))
        spectrum_loss = tf.math.square(tf.math.log(P_pred/P_truth)) 
        spectrum_loss = tf.cast(spectrum_loss, dtype=tf.float32)
        #sprectrum_loss = K.sum(spectrum_loss, axis = 1)
        return spectrum_loss
    return _power_spectrum

def encoder_gen(input_shape: tuple, encoder_config: dict, id):
    """
    Create the architecture for the VAE encoder. 
    """

    class EncoderResult():
        pass 

    encoder_result = EncoderResult()
    
    
    
    inputs = keras.layers.Input(shape=[input_shape, 1])
    print("shape of input after padding", inputs.shape)
    
    
    z = keras.layers.Flatten()(inputs)
    shape_before_flattening = K.int_shape(z) 
    print("shape of input after flattening", inputs.shape)
    
    
    print("shape after first Dense layer", z.shape)
    z = keras.layers.Dense(encoder_config["dense_1"]["dim"], activation=encoder_config["activation"])(z)
    print("shape after first Dense layer", z.shape)
    
    z = keras.layers.Dense(encoder_config["dense_2"]["dim"], activation=encoder_config["activation"])(z)
    print("shape after second Dense layer", z.shape)
    
    # Compute mean and log variance 
    z_mean = keras.layers.Dense(encoder_config["latent_dim"], name='z_mean')(z)
    z_log_var = keras.layers.Dense(encoder_config["latent_dim"], name='z_log_var')(z)

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
  
    return encoder_result, shape_before_flattening

def decoder_gen(
    original_input: tuple,
    decoder_config: dict, flatter_shape
):
    """
    Create the architecture for the VAE decoder 
    """
    
    decoder_inputs = keras.layers.Input(shape=[decoder_config["latent_dim"]])
    print("decoder_inputs", decoder_inputs._keras_shape)
    
    
    #x = keras.layers.Dense(np.prod(flatter_shape[1:]), activation=decoder_config["activation"])(decoder_inputs)
    #print("shape after initial change", x._keras_shape)
    
    # Reshape input to be an image 
    #x = keras.layers.Reshape(flatter_shape[1:])(x)
    #print("shape after resdhaping to an image", x._keras_shape)
    
    #x = keras.layers.Dense(decoder_config["dense_1"]["dim"], activation=decoder_config["activation"])(x)
    #print("shape after first dense layer", x._keras_shape)
    
    x = keras.layers.Dense(decoder_config["dense_1"]["dim"], activation=decoder_config["activation"])(decoder_inputs)
    print("shape after first dense layer", x._keras_shape)
    
    x = keras.layers.Dense(decoder_config["dense_2"]["dim"], activation=decoder_config["activation"])(x)
    print("shape after second dense layer", x.shape)

    x_mu = keras.layers.Dense(decoder_config["dense_mu"]["dim"], activation=decoder_config["dense_mu"]["activation"])(x)
    print("shape after dense mu layer", x_mu._keras_shape)

    x_log_var = keras.layers.Dense(decoder_config["dense_log_var"]["dim"], activation=decoder_config["dense_log_var"]["activation"])(x)
    print("shape after dense log var layer", x_log_var._keras_shape)
   
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
    
    #constraint_train_losses = hdict['_constrainer']
    #constraint_valid_losses = hdict['val__constrainer']

    total_train_losses = hdict['_kl_reconstruction_loss']
    total_valid_losses = hdict['val__kl_reconstruction_loss']

    epochs = range(1, len(train_reconstruction_losses) + 1)

    #fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12.8, 4.8))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.8, 4.8))

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

    img_height = train_data.shape[1]

    print("Image shape:", img_height)
    
    # Construct VAE Encoder 
    encoder_result, shape_flatten = encoder_gen((img_height), model_config["encoder"], args.id)
    # Construct VAE Decoder 
    vae_decoder = decoder_gen(
        (img_height),  
        model_config["decoder"], shape_flatten
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
    # Compile model
    vae.compile(
        # loss=reconstruction, 
        loss=kl_reconstruction_loss(
            encoder_result.z_log_var, 
            encoder_result.z_mean, 
            vae,
            stat_weight
        ), 
        optimizer=optimizer, 
        metrics=[
            reconstruction, 
            kl(
                encoder_result.z_log_var, 
                encoder_result.z_mean
            ), 
            kl_reconstruction_loss(
                encoder_result.z_log_var, 
                encoder_result.z_mean, 
                vae,
                stat_weight
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