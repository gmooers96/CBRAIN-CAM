import argparse 
import json 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import keras
from keras import layers
from keras import backend as K
import tensorflow as tf 

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE

from train_fully_conv import encoder_gen, decoder_gen, kl_reconstruction_loss
import numpy as np
import gc
import tensorflow_probability as tfp 

def f_norm(true, pred):
    covariance_truth = tfp.stats.covariance(true)
    covariance_prediction = tfp.stats.covariance(pred)
    covariance_truth = tf.cast(covariance_truth, dtype=tf.float32)
    f_dist = tf.norm(covariance_prediction-covariance_truth, ord="euclidean")
    return f_dist

def reconstruct_targets_paper(vae, test_data, targets, id, dataset_max, dataset_min):
    """
    TODO
    """
    original_samples = []
    recon_means = []
    recon_vars = []

    vmin = 1000
    vmax = -1

    vmin_var = 1000
    vmax_var = -1

    for target in targets:

        sample = test_data[target]
        sample_mean_var = vae.predict(np.expand_dims(sample, 0))
        sample_mean = sample_mean_var[0, :128*30]
        sample_log_var = sample_mean_var[0, 128*30:]

        # Sample reconstruction based on predicted mean and variance
        recon_mean = sample_mean
        recon_var = np.exp(sample_log_var)
        recon_sample = recon_mean + recon_var
        # recon_sample = np.random.multivariate_normal(sample_mean, np.exp(sample_log_var) * np.identity(128*30))
        
        # Rescale original sample and reconstruction to original scale
        sample = np.interp(sample, (0, 1), (dataset_min, dataset_max))
        recon_mean = np.interp(recon_mean, (0, 1), (dataset_min, dataset_max))
        recon_sample = np.interp(recon_sample, (0, 1), (dataset_min, dataset_max))
        recon_var = recon_sample - recon_mean

        # Get min and max of original and reconstructed 
        max_reconstructed = np.max(recon_mean)
        max_recon_var = np.max(recon_var)
        print("max of reconstructed", max_reconstructed)
        max_sample = np.max(sample.reshape((128*30,)))
        print("max of original", max_sample)
        min_reconstructed = np.min(recon_mean)
        min_recon_var = np.min(recon_var)
        print("min of reconstructed", min_reconstructed)
        min_sample = np.min(sample.reshape((128*30,)))
        print("min of original", min_sample)

        # Reshape reconstructed sample 
        recon_mean = recon_mean.reshape((30, 128))
        recon_var = recon_var.reshape((30, 128))

        original_samples.append(sample[:, :, 0])
        recon_means.append(recon_mean)
        recon_vars.append(recon_var)

        vmin = min(vmin, min_reconstructed, min_sample)
        vmax = max(vmax, max_reconstructed, max_sample)

        vmin_var = min(vmin_var, min_recon_var)
        vmax_var = max(vmax_var, max_recon_var)

    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 8
    plt.rcParams["figure.figsize"] = fig_size
    fig, axs = plt.subplots(len(targets), 2, sharex=True, sharey=True, constrained_layout=True)

    def fmt(x, pos):
        return "{:.2f}".format(x)

    for i in range(len(targets)): 
        y_ticks = np.arange(1400, 0, -400)
        #print("y ticks", y_ticks)

        sub_img = axs[i, 0].imshow(original_samples[i], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axs[i, 0].invert_yaxis()
        axs[i, 0].set_yticklabels(y_ticks)

        if i == 2:
            axs[i, 0].set_ylabel("Pressure (hPa)", fontsize=12, labelpad=10)
            
        sub_img = axs[i, 1].imshow(recon_means[i], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axs[i, 1].invert_yaxis()

        if i == 0:
            axs[i, 0].set_title("Original")
            axs[i, 1].set_title("VAE Reconstruction Mean")

        if i == len(targets) - 1:
            axs[i, 0].set_xlabel('CRMs', fontsize=12, labelpad=5)
            axs[i, 1].set_xlabel('CRMs', fontsize=12, labelpad=5)
            fig.colorbar(sub_img, ax=axs[:, 1], label="Vertical Velocity", shrink=0.6)
        #axs[i,1].set_yticks([])
        #if  i < len(targets) - 2:
            #axs[i, 0].set_xticks([])
            #axs[i, 1].set_xticks([])


    # Hide x labels and tick labels for all but bottom plot.
    for row in axs:
        for ax in row:
            ax.label_outer()

    plt.savefig('./model_graphs/reconstructions/Paper_target_test_reconstructions_{}.png'.format(id))


def reconstruct_targets(vae, test_data, targets, id, dataset_max, dataset_min):
    """
    TODO
    """
    original_samples = []
    recon_means = []
    recon_vars = []

    vmin = 1000
    vmax = -1

    vmin_var = 1000
    vmax_var = -1

    for target in targets:

        sample = test_data[target]
        sample_mean_var = vae.predict(np.expand_dims(sample, 0))
        sample_mean = sample_mean_var[0, :128*30]
        sample_log_var = sample_mean_var[0, 128*30:]

        # Sample reconstruction based on predicted mean and variance
        recon_mean = sample_mean
        recon_var = np.exp(sample_log_var)
        recon_sample = recon_mean + recon_var
        # recon_sample = np.random.multivariate_normal(sample_mean, np.exp(sample_log_var) * np.identity(128*30))
        
        # Rescale original sample and reconstruction to original scale
        sample = np.interp(sample, (0, 1), (dataset_min, dataset_max))
        recon_mean = np.interp(recon_mean, (0, 1), (dataset_min, dataset_max))
        recon_sample = np.interp(recon_sample, (0, 1), (dataset_min, dataset_max))
        recon_var = recon_sample - recon_mean

        # Get min and max of original and reconstructed 
        max_reconstructed = np.max(recon_mean)
        max_recon_var = np.max(recon_var)
        print("max of reconstructed", max_reconstructed)
        max_sample = np.max(sample.reshape((128*30,)))
        print("max of original", max_sample)
        min_reconstructed = np.min(recon_mean)
        min_recon_var = np.min(recon_var)
        print("min of reconstructed", min_reconstructed)
        min_sample = np.min(sample.reshape((128*30,)))
        print("min of original", min_sample)

        # Reshape reconstructed sample 
        recon_mean = recon_mean.reshape((30, 128))
        recon_var = recon_var.reshape((30, 128))

        original_samples.append(sample[:, :, 0])
        recon_means.append(recon_mean)
        recon_vars.append(recon_var)

        vmin = min(vmin, min_reconstructed, min_sample)
        vmax = max(vmax, max_reconstructed, max_sample)

        vmin_var = min(vmin_var, min_recon_var)
        vmax_var = max(vmax_var, max_recon_var)

    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 8
    plt.rcParams["figure.figsize"] = fig_size
    fig, axs = plt.subplots(len(targets), 3, sharex=True, sharey=True, constrained_layout=True)

    def fmt(x, pos):
        return "{:.2f}".format(x)

    for i in range(len(targets)): 
        y_ticks = np.arange(1800, 0, -800)
        print("y ticks", y_ticks)

        sub_img = axs[i, 0].imshow(original_samples[i], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axs[i, 0].invert_yaxis()
        axs[i, 0].set_yticklabels(y_ticks)

        if i == 2:
            axs[i, 0].set_ylabel("Pressure (mbs)", fontsize=12, labelpad=10)
            
        sub_img = axs[i, 1].imshow(recon_means[i], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axs[i, 1].invert_yaxis()

        sub_img_var = axs[i, 2].imshow(recon_vars[i], cmap='RdBu_r', vmin=vmin_var, vmax=vmax_var)
        axs[i, 2].invert_yaxis()

        if i == 0:
            axs[i, 0].set_title("Original")
            axs[i, 1].set_title("Reconstruction Mean")
            axs[i, 2].set_title("Reconstruction Variance")

        if i == len(targets) - 1:
            axs[i, 0].set_xlabel('CRMs', fontsize=12, labelpad=5)
            axs[i, 1].set_xlabel('CRMs', fontsize=12, labelpad=5)
            axs[i, 2].set_xlabel('CRMs', fontsize=12, labelpad=5)
            fig.colorbar(sub_img, ax=axs[:, 1], label="Vertical Velocity", shrink=0.6)
            cb = fig.colorbar(sub_img_var, ax=axs[:, 2], shrink=0.6, format=ticker.FuncFormatter(fmt))
            cb.set_label("Variance", labelpad=10)


    # Hide x labels and tick labels for all but bottom plot.
    for row in axs:
        for ax in row:
            ax.label_outer()

    plt.savefig('./model_graphs/reconstructions/target_test_reconstructions_{}.png'.format(id))

def sample_reconstructions(vae, train_data, test_data, id, dataset_max, dataset_min): 
    """
    TODO 
    """
 
    original_samples = []
    recon_samples = []

    min_max = []

    for i in range(5):
        rand_sample = np.random.randint(0, len(train_data))

        sample = train_data[rand_sample]
        sample_mean_var = vae.predict(np.expand_dims(sample, 0))
        sample_mean = sample_mean_var[0, :128*30]
        sample_log_var = sample_mean_var[0, 128*30:]

        recon_sample = sample_mean
        
        sample = np.interp(sample, (0, 1), (dataset_min, dataset_max))
        recon_sample = np.interp(recon_sample, (0, 1), (dataset_min, dataset_max))

        print("original sample", sample.reshape((128*30,)))
        print("reconstructed sample", recon_sample)
        print(np.max(np.abs(sample.reshape((128*30,)) - recon_sample)))
        max_reconstructed = np.max(np.abs(recon_sample))
        print("max of reconstructed", max_reconstructed)
        max_sample = np.max(sample.reshape((128*30,)))
        print("max of original", max_sample)
        min_reconstructed = np.min(recon_sample)
        print("min of reconstructed", min_reconstructed)
        min_sample = np.min(sample.reshape((128*30,)))
        print("min of original", min_sample)
        recon_sample = recon_sample.reshape((30, 128))

        original_samples.append(sample[:, :, 0])
        recon_samples.append(recon_sample)

        min_max.append((min(min_reconstructed, min_sample), max(max_reconstructed, max_sample)))

    fig, axs = plt.subplots(5, 2)

    for i in range(5): 
        vmin = min_max[i][0]
        vmax = min_max[i][1]
    
        sub_img = axs[i, 0].imshow(original_samples[i], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axs[i, 0].set_ylim(axs[i, 0].get_ylim()[::-1])
        fig.colorbar(sub_img, ax=axs[i, 0])

        sub_img = axs[i, 1].imshow(recon_samples[i], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axs[i, 1].set_ylim(axs[i, 1].get_ylim()[::-1])
        fig.colorbar(sub_img, ax=axs[i, 1])

    plt.savefig('./model_graphs/reconstructions/reconstructed_train_samples_{}.png'.format(id))

def sample_latent_space(vae_encoder, train_data, test_data, id, dataset_min, dataset_max, test_labels, dataset_type): 
    """
    Create a scatter plot of the latent space containing all test samples.
    """

    # Predict latent train & test data
    #print(test_data.shape)
    _, _, z_test = vae_encoder.predict(test_data)
    _, _, z_train = vae_encoder.predict(train_data)
    #print("This is z_test")
    #print(z_test.shape)
    # Apply scaling and tsne 
    sc = StandardScaler()
    z_train_std = sc.fit_transform(z_train)
    
    z_test_std = sc.transform(z_test)
    # Instantiate PCA 
    pca = PCA(n_components=32)
    pca.fit(z_train_std)

    z_test_pca = pca.transform(z_test_std)
    # Instantiate TSNE
    tsne = TSNE(n_components=2)

    z_test_tsne = tsne.fit_transform(z_test_pca)
    np.save("/fast/gmooers/gmooers_git/CBRAIN-CAM/MAPS/Saved_Data/Other_Day_Latent_Space__{}".format(id), z_test_tsne)
    if dataset_type == "half_deep_convection":
        colors = ["#FF4940", "#3D9AD1"]
        print("made it here")
        convection = np.squeeze(z_test_tsne[np.where(test_labels == 0),:])
        no_convection = np.squeeze(z_test_tsne[np.where(test_labels == 1),:])
        #fake = np.squeeze(z_test_tsne[np.where(test_labels == 2),:])
        plt.scatter(x=convection[:, 0], y=convection[:, 1], c="#FF4940", s=0.4, label="No Convective Activity")
        plt.scatter(x=no_convection[:, 0], y=no_convection[:, 1], c="#3D9AD1", s=0.4, label="Convective Activity")
        #plt.scatter(x=fake[:, 0], y=fake[:, 1], c="yellow", s=0.4, label="Blue Noise")
        plt.legend()

    else:
        #plt.scatter(x=z_test_tsne[:, 0], y=z_test_tsne[:, 1], c=test_labels, s=1)
        plt.scatter(x=z_test_tsne[:, 0], y=z_test_tsne[:, 1], s=0.1)
        plt.colorbar()

    plt.savefig('./model_graphs/latent_space/Other_Day_binary_latent_space_with_pca_{}.png'.format(id))


def sample_latent_space_var(vae_encoder, train_data, test_data, id, dataset_min, dataset_max, test_labels, dataset_type): 
    """
    Create a scatter plot of the latent space containing all test samples.
    """

    # Predict latent train & test data
    test_mean, test_log_var, z_test = vae_encoder.predict(test_data)
    train_mean, train_log_var, z_train = vae_encoder.predict(train_data)
    #np.save("Saved_Data/Test_Z_Samples.npy", z_test)
    #np.save("Saved_Data/Test_Mean_Samples.npy", test_mean)
    #np.save("Saved_Data/Test_Log_Var_Samples.npy", test_log_var)
    #print(gdfgdgdsgdsfgdfg)
    train_mean_var = np.concatenate((train_mean, train_log_var), axis=1)
    test_mean_var = np.concatenate((test_mean, test_log_var), axis=1)
    #np.save("Saved_Data/Train_High_Dim_Latent_Space.npy", train_mean_var)
    #np.save("Saved_Data/Test_High_Dim_Latent_Space.npy", test_mean_var)
    # Apply scaling and tsne 
    sc = StandardScaler()
    z_train_std = sc.fit_transform(train_mean_var)
    #z_train_std = sc.fit_transform(train_log_var)
    
    z_test_std = sc.transform(test_mean_var)
    #z_test_std = sc.transform(test_log_var)
    # Instantiate PCA 
    pca = PCA(n_components=32)
    pca.fit(z_train_std)

    z_test_pca = pca.transform(z_test_std)
    # Instantiate TSNE
    tsne = TSNE(n_components=2)

    z_test_tsne = tsne.fit_transform(z_test_pca)
    np.save("/fast/gmooers/gmooers_git/CBRAIN-CAM/MAPS/Saved_Data/Trial_Amazon_Diurnal_Mean_Var_Latent_Space__{}".format(id), z_test_tsne)
    if dataset_type == "half_deep_convection":
        colors = ["#FF4940", "#3D9AD1"]
        print("made it here")
        convection = np.squeeze(z_test_tsne[np.where(test_labels == 0),:])
        no_convection = np.squeeze(z_test_tsne[np.where(test_labels == 1),:])
        #fake = np.squeeze(z_test_tsne[np.where(test_labels == 2),:])
        plt.scatter(x=convection[:, 0], y=convection[:, 1], c="#FF4940", s=0.4, label="No Convective Activity")
        plt.scatter(x=no_convection[:, 0], y=no_convection[:, 1], c="#3D9AD1", s=0.4, label="Convective Activity")
        #plt.scatter(x=fake[:, 0], y=fake[:, 1], c="yellow", s=0.4, label="Blue Noise")
        plt.legend()

    else:
        #plt.scatter(x=z_test_tsne[:, 0], y=z_test_tsne[:, 1], c=test_labels, s=1)
        plt.scatter(x=z_test_tsne[:, 0], y=z_test_tsne[:, 1], s=0.1)
        plt.colorbar()

    plt.savefig('./model_graphs/latent_space/Trial_Amazon_Diurnal_Mean_Var_latent_space_with_pca_{}.png'.format(id))    

    
    
def sample_frob_norm(vae, decoder, vae_encoder, train_data, test_data, id, dataset_min, dataset_max, test_labels, dataset_type): 
    """
    Create a scatter plot of the latent space containing all test samples.
    """

    # Predict latent train & test data
    test_mean, test_log_var, z_test = vae_encoder.predict(test_data)
    print("made it here")
    sample_mean_var = decoder.predict(z_test) 
    sample_mean = sample_mean_var[:, :128*30]
    truths = np.reshape(test_data, (len(test_data),30*128))
    

    Rough_Metric = f_norm(truths, sample_mean)
    
    sess = tf.InteractiveSession()
    RM = Rough_Metric.eval()
    gc.collect()
    print(RM.shape)
    print(RM)
    np.save("Saved_Data/Rough_Overall_FR_Norm__{}.npy".format(id), RM)
    print("completed")      
    
def sample_anon_detect(vae, decoder, vae_encoder, train_data, test_data, id, dataset_min, dataset_max, test_labels, dataset_type): 
    """
    Create a scatter plot of the latent space containing all test samples.
    """

    # Predict latent train & test data
    test_mean, test_log_var, z_test = vae_encoder.predict(test_data)
    #train_mean, train_log_var, z_train = vae_encoder.predict(train_data)
    print("the shape of the test data is",test_data.shape)
    print("the shape of test mean is",test_mean.shape)
    print("the shape of test log var is",test_log_var.shape)
    print("the shape of z test is",z_test.shape)
    print("made it here")
    sample_mean_var = decoder.predict(z_test) 
    sample_mean = sample_mean_var[:, :128*30]
    sample_log_var = sample_mean_var[:, 128*30:]
    print("the shape of the decoder output is", sample_mean_var.shape)
    print("the shape of the sample mean is", sample_mean.shape)
    print("The shape of the sample log var is", sample_log_var.shape)
    print("hello there")
    losses = kl_reconstruction_loss(test_log_var, test_mean, vae)
    truths = np.reshape(test_data, (len(test_data),30*128))
    print("The reshaped test data is", truths.shape)
    something = losses(truths,sample_mean_var)
    sess = tf.InteractiveSession()
    elbo = something.eval()
    gc.collect()
    print(elbo.shape)
    np.save("Saved_Data/50_50_Centered_31_elbo.npy",elbo)
    print("completed")
    
    
def generate_samples(decoder, dataset_min, dataset_max, latent_dim: int, id):
    """
    Sample points from prior and send through decoder to get 
    sample images.
    """
    # sample from prior 
    num_samples = 3
    z = np.random.normal(size=(num_samples, latent_dim))

    # Get output from decoder 
    sample_mean_var = decoder.predict(z)

    # Extract mean and variance 
    sample_mean = sample_mean_var[:, :128*30]
    sample_log_var = sample_mean_var[:, 128*30:]

    fig, axs = plt.subplots(num_samples, 1)

    recon_samples = []
    for i in range(num_samples):
        print(sample_mean[i])
        print(sample_log_var[i])
        print(sample_mean[i].shape)
        # Sample from gaussian decoder outputs 
        recon_sample = np.random.multivariate_normal(sample_mean[i], np.exp(sample_log_var[i]) * np.identity(128*30))

        # Unnormalize sample 
        recon_sample = np.interp(recon_sample, (0, 1), (dataset_min, dataset_max))

        # Reshape
        recon_sample = recon_sample.reshape((30, 128))

        recon_samples.append(recon_sample)

    vmin = np.min(recon_samples)
    vmax = np.max(recon_samples)
    print(vmin, vmax)
    for i in range(num_samples):
        # Show image
        sub_img = axs[i].imshow(recon_samples[i], cmap='coolwarm', vmin=vmin, vmax=vmax)
        fig.colorbar(sub_img, ax=axs[i])

        # Flip y-axis
        axs[i].set_ylim(axs[i].get_ylim()[::-1])
        
    # fig.colorbar(sub_img, ax=axs)
    plt.tight_layout()
    #plt.suptitle("Generated W Fields")
    plt.savefig('./model_graphs/generated/generated_samples_{}.png'.format(id))



def main():
    args = argument_parsing()
    print("Command line args:", args)

    f = open("./model_config/config_{}.json".format(args.id))
    model_config = json.load(f)
    f.close()

    train_data = np.load(model_config["data"]["training_data_path"])
    test_data = np.load(model_config["data"]["test_data_path"])

    # test_labels = np.load(model_config["data"]["test_labels"])[:, 0, 0]
    test_labels = np.load(model_config["data"]["test_labels"])
    print("Test labels shape:", test_labels.shape, model_config["data"]["test_labels"])

    dataset_max = np.load(model_config["data"]["max_scalar"])
    dataset_min = np.load(model_config["data"]["min_scalar"])

    print("dataset max", dataset_max)
    print("dataset min", dataset_min)

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

    _, _, z = encoder_result.vae_encoder(encoder_result.inputs)
    x_mu_var = vae_decoder(z)
    vae = keras.Model(inputs=[encoder_result.inputs], outputs=[x_mu_var])

    # load weights from file
    vae.load_weights('./models/model_{}.th'.format(args.id))
    print("weights loaded")

    train_data = train_data.reshape(train_data.shape+(1,))
    test_data = test_data.reshape(test_data.shape+(1,))

    # get side by side plots of original vs. reconstructed
    sample_frob_norm(vae, vae_decoder, encoder_result.vae_encoder, train_data, test_data, args.id, dataset_min, dataset_max, test_labels, args.dataset_type)
    #reconstruct_targets_paper(vae, test_data, [2, 15, 66 , 85, 94], args.id, dataset_max, dataset_min)
    #sample_reconstructions(vae, train_data, test_data, args.id, dataset_max, dataset_min)
    #reconstruct_targets(vae, test_data, [2, 15, 66 , 85, 94], args.id, dataset_max, dataset_min)
    #sample_latent_space(encoder_result.vae_encoder, train_data, test_data, args.id, dataset_min, dataset_max, test_labels, args.dataset_type)
    #sample_latent_space_var(encoder_result.vae_encoder, train_data, test_data, args.id, dataset_min, dataset_max, test_labels, args.dataset_type)
    #sample_anon_detect(vae, vae_decoder, encoder_result.vae_encoder, train_data, test_data, args.id, dataset_min, dataset_max, test_labels, args.dataset_type)
    #generate_samples(vae_decoder, dataset_min, dataset_max, model_config["encoder"]["latent_dim"], args.id)

def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, help='This option specifies the id of the config file to use to train the VAE.')
    parser.add_argument('--dataset_type', type=str, help='Name of the dataset that model was trained on.')

    args = parser.parse_args()
    return args 

if __name__ == "__main__":
    main()
