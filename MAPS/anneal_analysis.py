import argparse 
import json 

import numpy as np
import matplotlib.pyplot as plt

import keras
from keras import layers
from keras import backend as K
from scipy.stats import norm
#from train_vae import encoder_gen, decoder_gen
from kl_anneal_vae_train import encoder_gen, decoder_gen
        
def spectrum_gen(h, dx):
    nx = len(h)
    #Get half the length of the series to avoid redudant information
    npositive = nx//2
    pslice = slice(1, npositive)
    #Get frequencies
    freqs = np.fft.fftfreq(nx, d=dx)[pslice] 
    #perform the fft 
    ft = np.fft.fft(h)[pslice]
    #remove imaginary componant of the fft and square
    psraw = np.conjugate(ft) *ft
    #double to account for the negative half that was removed above
    psraw *= 2.0
    #Normalization for the power spectrum
    psraw /= nx**2
    #Go from the Power Spectrum to Power Density
    psdraw = psraw * dx * nx
    return freqs, psraw, psdraw


def spectrum_generator(targets, features, levels, time_space):
    targ_freqs, targ_psraw, targ_psdraw = spectrum_gen(np.squeeze(targets[1,:]), time_space)
    depth = len(targ_psdraw)
    target_collector = np.zeros(shape=(levels, depth))
    target_collector[:,:] = np.nan
    feature_collector = np.zeros(shape=(levels, depth))
    feature_collector[:,:] = np.nan
    counter = 0
    for i in range(levels):
            target = np.squeeze(targets[i, :])
            feature = np.squeeze(features[i, :])
            targ_freqs, targ_psraw, targ_psdraw = spectrum_gen(target, time_space)
            feat_freqs, feat_psraw, feat_psdraw = spectrum_gen(feature, time_space)
            target_collector[i, :] = targ_psdraw
            feature_collector[i, :] = feat_psdraw
    rep_target = np.nanmean(target_collector, axis = 0)
    rep_pred = np.nanmean(feature_collector, axis = 0)
    #plotter(rep_target, rep_pred, targ_freqs, "Image average Signal")
    #return targ_freqs, feat_freqs, target_collector, feature_collector
    return rep_target, rep_pred, targ_freqs

def mse_metric(p, q):
    mse = np.square(np.subtract(p,q)).mean()
    return mse

def pdf_gen(dist):
    mu, std = norm.fit(dist)
    #print("this is the dist", dist.shape)
    if dist.ndim > 2:
        dist = np.reshape(dist, (len(dist)*len(dist[0]),len(dist[0][0])))
    plt.hist(dist, bins=25, density=True, alpha=0.6)
    #print("Graphed")
    xmin, xmax = plt.xlim()
    #print("limits")
    x = np.linspace(xmin, xmax, len(dist))
    #print("linspace")
    pdf = norm.pdf(x, mu, std)
    #print("made it to pdf func end")
    return pdf

def Hellinger_Dist(p, q):
    p = pdf_gen(p)
    q = pdf_gen(q)
    hd = np.sqrt(np.sum((np.sqrt(p.ravel()) - np.sqrt(q.ravel())) ** 2)) / np.sqrt(2)
    return hd

def sample_reconstructions(decoder, encoder, vae, train_data, test_data, id): 
    """
    TODO 
    """

    # get random sample 
    #x_test_encoded = np.array(encoder.vae_encoder.predict(test_data, batch_size = 128))
    #plt.figure(figsize=(6,8))
    #plt.scatter(x_test_encoded[:, 0], x_test_encoded[:,1])
    #y_test = np.load('/fast/gmooers/Preprocessed_Data/W_100_X/Y_Land_Sea_Test.npy')
    #plt.scatter(x_test_encoded[2, :,0], x_test_encoded[2, :,1], c = y_test)
    #plt.colorbar()
    #plt.title("Latent Space Representation")
    #plt.savefig('./model_graphs/land_latent_space_{}.png'.format(id))
    #plt.close()
    
    #plt.figure(figsize=(6,8))
    #plt.scatter(x_test_encoded[:, 0], x_test_encoded[:,1])
    #y_test = np.load('/fast/gmooers/Preprocessed_Data/W_100_X/Y_Convection_Test.npy')
    #plt.scatter(x_test_encoded[2, :,0], x_test_encoded[2, :,1], c = y_test)
    #plt.colorbar()
    #plt.title("Latent Space Representation")
    #plt.savefig('./model_graphs/Deep_Shallow_latent_space_{}.png'.format(id))
    #plt.close()
    
    #plt.figure(figsize=(6,8))
    #plt.scatter(x_test_encoded[:, 0], x_test_encoded[:,1])
    #y_test = np.load('/fast/gmooers/Preprocessed_Data/W_Big_Half_Deep_Convection/Y_Test.npy')
    #plt.scatter(x_test_encoded[2, :,0], x_test_encoded[2, :,1], c = y_test[:,0,0])
    #plt.colorbar()
    #plt.title("Latent Space Representation")
    #plt.savefig('./model_graphs/W_Conv_latent_space_{}.png'.format(id))
    #plt.close()
    
    #plt.figure(figsize=(6,8))
    #plt.scatter(x_test_encoded[:, 0], x_test_encoded[:,1])
    #y_test = np.load('/fast/gmooers/Preprocessed_Data/W_Big_Half_Deep_Convection/Y_Test.npy')
    #plt.scatter(x_test_encoded[2, :,0], x_test_encoded[2, :,1])
    #plt.colorbar()
    #plt.title("Latent Space Representation")
    #plt.savefig('./model_graphs/Latent_Shape_{}.png'.format(id))
    #plt.close()
    
    original_samples = []
    recon_samples = []
    hds = []
    mses = []
    samples = 3
    ##################################################################################
    #begin generation code
#     for i in range(samples):
#         z_new = np.random.normal(size=(1,2))
#         new_image = decoder.predict(z_new)
#         new_image=np.squeeze(new_image)
#         sample_mean = new_image[:128*30]
#         sample_log_var = new_image[128*30:]
#         recon_sample = np.random.multivariate_normal(sample_mean, np.exp(sample_log_var) * np.identity(128*30))
#         print("made it past recon sample")
#         new_image = recon_sample.reshape((30,128))
#         recon_samples.append(new_image)
        
#     Max_Scalar = np.load('/fast/gmooers/Preprocessed_Data/W_100_X/Space_Time_Max_Scalar.npy')
#     Min_Scalar = np.load('/fast/gmooers/Preprocessed_Data/W_100_X/Space_Time_Min_Scalar.npy')
#     Unscaled_Predict_Images = np.interp(recon_samples, (0, 1), (Min_Scalar, Max_Scalar))
#     fig, axs = plt.subplots(samples)
#     max_val = np.max(Unscaled_Predict_Images)
#     min_val = np.min(Unscaled_Predict_Images)
    
#     for i in range(samples): 
#         cb = axs[i].imshow(Unscaled_Predict_Images[i], vmax = max_val, vmin = min_val, cmap='RdBu_r')
#         axs[i].invert_yaxis()
#         y_ticks = np.arange(1400, 0, -400)
#         axs[i].set_yticklabels(y_ticks)
#         if i == 0:
#             axs[i].set_title("Generations")
#             cbaxes = fig.add_axes([0.908, 0.1, 0.03, 0.8])
#             fig.colorbar(cb, cax = cbaxes, label = "Vertical Velocity")
    
#         if i < samples-1:
#             axs[i].set_xticks([])
#         if i == samples-1:
#             label = axs[i].set_xlabel('CRMs', fontsize = 12)
#             axs[i].xaxis.set_label_coords(-0.05, -4.825)
           
#         label = axs[i].set_ylabel('Pressure (mbs)', fontsize = 12)
#         axs[i].yaxis.set_label_coords(-1.22, -1.525)
              
 
    
    
#     plt.subplots_adjust(wspace=0.05, hspace=0)
#     plt.suptitle("VAE Samples")
#     plt.savefig('./model_graphs/generated_samples_{}.png'.format(id))
#     plt.close()
    #end generation code
    ##################################################################################
    original_samples = []
    recon_samples = []
    spectrum_pred = []
    spectrum_truth = []
    samples = 3
    fig, ax = plt.subplots(samples,1)
    for i in range(samples):
        rand_sample = np.random.randint(0, len(test_data))

        sample = test_data[rand_sample]
        sample_mean_var = vae.predict(np.expand_dims(sample, 0))
        sample_mean = sample_mean_var[0, :128*30]
        sample_log_var = sample_mean_var[0, 128*30:]
        print("made it here")
        recon_sample = np.random.multivariate_normal(sample_mean, np.exp(sample_log_var) * np.identity(128*30))
        print("made it past recon sample")
        recon_sample = recon_sample.reshape((30, 128))

        original_samples.append(sample[:, :, 0])
        recon_samples.append(recon_sample)
        h = Hellinger_Dist(np.array(sample[:, :, 0]), np.array(recon_sample))
        hds.append(h)
        mse = mse_metric(np.array(sample[:, :, 0]), np.array(recon_sample))
        mses.append(mse)
        #rep_target, rep_pred, targ_freqs = spectrum_generator(sample[:, :, 0], recon_sample, 30, 1/128)
        rep_target, rep_pred, targ_freqs = spectrum_generator(sample[:, :, 0], recon_sample, 30, 1)
        spectrum_pred.append(rep_pred)
        spectrum_truth.append(rep_target)
        #ax[i].plot(targ_freqs, rep_target, label = "Original")
        #ax[i].plot(targ_freqs, rep_pred, label = "Reconstruction")
        ax[i].plot(1/targ_freqs, rep_target, label = "Original")
        ax[i].plot(1/targ_freqs, rep_pred, label = "Reconstruction")
        ax[i].set_yscale('log')
        ax[i].set_xscale('log')
        if i == 0:
            ax[i].legend()
        if i < samples-1:
            ax[i].set_xticks([])
        if i == samples-1:
            ax[i].set_xlabel("CRM Spacing")
        ax[i].set_ylabel(r'$\frac{m^2*crm}{s^2}$')
    
    plt.suptitle("Spatial Spectral Analysis")
    plt.savefig('./model_graphs/fft_{}.png'.format(id))
    plt.close()
    
    overall_truth = np.nanmean(np.array(spectrum_truth),axis = 0)
    overall_pred = np.nanmean(np.array(spectrum_pred),axis = 0)
    #plt.plot(targ_freqs, overall_truth, label="Original")
    #plt.plot(targ_freqs, overall_pred, label="Reconstruction")
    plt.plot(1/targ_freqs, overall_truth, label="Original")
    plt.plot(1/targ_freqs, overall_pred, label="Reconstruction")
    plt.legend()
    plt.xlabel("CRM Spacing")
    plt.ylabel(r'$\frac{m^2*crm}{s^2}$')
    plt.yscale('log')
    plt.xscale('log')
    plt.title("Overall signal")
    plt.savefig('./model_graphs/overall_fft_{}.png'.format(id))
    plt.close()
    
    #Formula from Gagne et. al 2020
    hd = Hellinger_Dist(np.array(original_samples), np.array(recon_samples))
    
    Max_Scalar = np.load('/fast/gmooers/Preprocessed_Data/W_100_X/Space_Time_Max_Scalar.npy')
    Min_Scalar = np.load('/fast/gmooers/Preprocessed_Data/W_100_X/Space_Time_Min_Scalar.npy')
    Unscaled_Predict_Images = np.interp(recon_samples, (0, 1), (Min_Scalar, Max_Scalar))
    Unscaled_Test_Images = np.interp(original_samples, (0, 1), (Min_Scalar, Max_Scalar))
    fig, axs = plt.subplots(samples, 2)
    maxs = np.zeros(shape=(2))
    maxs[0] = np.max(Unscaled_Test_Images)
    maxs[1] = np.max(Unscaled_Predict_Images)
    max_val = np.max(maxs)

    mins = np.zeros(shape=(2))
    mins[0] = np.min(Unscaled_Test_Images)
    mins[1] = np.min(Unscaled_Predict_Images)
    min_val = np.min(mins)
    #np.save('/fast/gmooers/gmooers_git/CBRAIN-CAM/MAPS/Temp_Data/recon_sample.npy', recon_samples)
    #np.save('/fast/gmooers/gmooers_git/CBRAIN-CAM/MAPS/Temp_Data/original_sample.npy', original_samples)
    count_a = 0
    count_b = 0
    count_c = 0
    for i in range(samples*2): 
        if i % 2 == 0:
            cb = axs[int(i/2), 0].imshow(Unscaled_Test_Images[int(i/2)], vmax = max_val, vmin = min_val, cmap='RdBu_r')
            axs[int(i/2), 0].invert_yaxis()
            y_ticks = np.arange(1800, 0, -800)
            axs[int(i/2), 0].set_yticklabels(y_ticks)
            if count_b == 0:
                axs[int(i/2), 0].set_title("Originals")
                count_b = 1
            if i == 0:
                cbaxes = fig.add_axes([0.908, 0.1, 0.03, 0.8])
                fig.colorbar(cb, cax = cbaxes, label = "Vertical Velocity")
            
            
        elif i % 2 == 1:
            axs[int(i/2), 1].imshow(Unscaled_Predict_Images[int(i/2)], vmax = max_val, vmin = min_val, cmap='RdBu_r')
            axs[int(i/2), 1].set_yticks([])
            axs[int(i/2), 1].invert_yaxis()
            if count_c == 0:
                axs[int(i/2), 1].set_title("Reconstructions")
                count_c = 1
            axs[int(i/2), 1].text(100, 25, "MSE: "+str(mses[int(i/2)])[:3]+str(mses[int(i/2)])[-4:], fontsize = 5)
            axs[int(i/2), 1].text(108, 20, "HD: "+str(hds[int(i/2)])[:4], fontsize = 5)
        if i < samples*2-3:
            axs[int(i/2), 1].set_xticks([])
            axs[int(i/2), 0].set_xticks([])
            if count_a == 0:
                label = axs[int(i/2), 1].set_xlabel('CRMs', fontsize = 12)
                axs[int(i/2), 1].xaxis.set_label_coords(-0.05, -4.825)
                label = axs[int(i/2), 1].set_ylabel('Pressure (mbs)', fontsize = 12)
                axs[int(i/2), 1].yaxis.set_label_coords(-1.22, -1.525)
                count_a = 1
 
    
    
    plt.subplots_adjust(wspace=0.05, hspace=0)
    plt.suptitle("VAE Samples")
    plt.savefig('./model_graphs/reconstructed_test_samples_{}.png'.format(id))
    plt.close()
    
    max_val = np.max(Unscaled_Predict_Images)
    min_val = np.min(Unscaled_Predict_Images)
    fig, axs = plt.subplots(samples)
    for i in range(samples):
        axs[i].imshow(Unscaled_Predict_Images[i], vmax = 1.5, vmin = min_val, cmap='RdBu_r')
        axs[i].set_yticklabels(y_ticks)
        axs[i].invert_yaxis()
        if i == samples-1:
            label = axs[i].set_xlabel('CRMs', fontsize = 12)
            axs[i].xaxis.set_label_coords(0.55, -0.325)
            label = axs[i].set_ylabel('Pressure (mbs)', fontsize = 12)
            axs[i].yaxis.set_label_coords(-0.12, 1.525)
        y_ticks = np.arange(1300, 100, -300)
        
    
    plt.subplots_adjust(wspace=0.05, hspace=0)
    plt.suptitle("VAE Comparison")
    plt.savefig('./model_graphs/reconstruct_only_{}.png'.format(id))
    plt.close()
    
    test = Unscaled_Test_Images.ravel()
    reconstruction = Unscaled_Predict_Images.ravel()
    shared_bins = np.histogram_bin_edges(reconstruction, bins=100)
    freq, edges = np.histogram(reconstruction, bins = shared_bins)
    freq_targ, edges_targ = np.histogram(test, bins = shared_bins)
    fig, ax = plt.subplots()
    
    plt.plot(edges[:-1], freq,  label = "Reconstruction", alpha = 0.5, color = 'blue')
    plt.plot(edges_targ[:-1], freq_targ, label = "Truth", alpha = 0.5, color = 'green')

    plt.xlabel('Vertical Velocity', fontsize = 15)
    plt.ylabel('Frequency', fontsize = 15)
    plt.title('Histogram.  Hellinger Distance: '+str(hd)[:4], fontsize = 15)
    plt.legend(loc = 'best')
    plt.savefig('./model_graphs/reconstructed_distribution_{}.png'.format(id))
    plt.close()


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
    encoder_result = encoder_gen((img_width, img_height), model_config["encoder"])

    # Construct VAE Decoder 
    vae_decoder = decoder_gen(
        (img_width, img_height),  
        model_config["decoder"],
        encoder_result.shape_before_flattening
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
    sample_reconstructions(vae_decoder, encoder_result, vae, train_data, test_data, args.id)

def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, help='This option specifies the id of the config file to use to train the VAE.')

    args = parser.parse_args()
    return args 

if __name__ == "__main__":
    main()