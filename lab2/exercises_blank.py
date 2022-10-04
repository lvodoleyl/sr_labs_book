# Exercises in order to perform laboratory work


# Import of modules
import numpy as np
import scipy.signal

from skimage.morphology import opening, closing


def load_vad_markup(path_to_rttm, signal, fs):
    # Function to read rttm files and generate VAD's markup in samples
    
    vad_markup = np.zeros(len(signal)).astype('float32')
        
    ###########################################################
    sec2num = lambda sec: int(round(float(sec)*fs))
    with open(path_to_rttm, 'r') as f:
        rttm = f.readlines()
    for line in rttm:
        _, _, _, start, duration, *other = line.replace('\n', '').split(' ')
        vad_markup[sec2num(start):sec2num(start)+sec2num(duration)] = 1
    ###########################################################
    
    return vad_markup

def framing(signal, window=320, shift=160):
    # Function to create frames from signal
    
    # shape   = (int((signal.shape[0] - window)/shift + 1), window)
    # frames  = np.zeros().astype('float32')

    ###########################################################
    signal_length = len(signal)
    num_frames = int(
        np.ceil(float(np.abs(signal_length - window)) / shift))

    pad_signal_length = num_frames * shift + window
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal, z)

    frames = np.array([pad_signal[idx:idx + window]
                       for idx in range(0, signal_length - shift - 1, shift)])
    
    ###########################################################
    
    return frames

def frame_energy(frames):
    # Function to compute frame energies
    
    E = np.zeros(frames.shape[0]).astype('float32')

    ###########################################################
    E = frames.sum(axis=1) # np.square(frames).sum(axis=1)
    ###########################################################
    
    return E

def norm_energy(E):
    # Function to normalize energy by mean energy and energy standard deviation
    
    E_norm = np.zeros(len(E)).astype('float32')

    ###########################################################
    E_norm = (E - E.mean()) / (E.var() ** 0.5)
    
    ###########################################################
    
    return E_norm

def gmm_train(E, gauss_pdf, n_realignment=10):
    # Function to train parameters of gaussian mixture model
    
    # Initialization gaussian mixture models
    w     = np.array([ 0.33, 0.33, 0.33])
    m     = np.array([-1.00, 0.00, 1.00])
    sigma = np.array([ 1.00, 1.00, 1.00])
    count_frames = len(E)
    g = np.zeros([len(E), len(w)])
    for n in range(n_realignment):
        ...
        # E-step
        ###########################################################
        eStep_get_g_i = lambda e: (w * gauss_pdf(e, m, sigma)) / (w * gauss_pdf(e, m, sigma)).sum()
        g = np.array(list(map(eStep_get_g_i, E)))
        ###########################################################

        # M-step
        ###########################################################
        w = g.sum(axis=0) / count_frames
        m = (g*np.repeat([E], 3, axis=0).T).sum(axis=0) / (count_frames * w)
        sigma = np.array([(g.T[idx]*(E - m[idx])**2).sum() / (count_frames * w[idx]) for idx in range(3)])
        sigma = np.sqrt(sigma)
        ###########################################################
        
    return w, m, sigma

def eval_frame_post_prob(E, gauss_pdf, w, m, sigma):
    # Function to estimate a posterior probability that frame isn't speech

    g0 = np.zeros(len(E))

    ###########################################################
    #for idx in range():
    #    g0[idx] = (w[0]*gauss_pdf(E[idx], m[0], sigma[0])) / (w*gauss_pdf(E[idx], m, sigma)).sum()
    g0 = np.array([(w[0]*gauss_pdf(E[idx], m[0], sigma[0])) / (w*gauss_pdf(E[idx], m, sigma)).sum()
                   for idx in range(len(E))])
    ###########################################################
            
    return g0

def energy_gmm_vad(signal, window, shift, gauss_pdf, n_realignment, vad_thr, mask_size_morph_filt):
    # Function to compute markup energy voice activity detector based of gaussian mixtures model
    
    # Squared signal
    squared_signal = signal**2
    
    # Frame signal with overlap
    frames = framing(squared_signal, window=window, shift=shift)
    
    # Sum frames to get energy
    E = frame_energy(frames)
    
    # Normalize the energy
    E_norm = norm_energy(E)
    
    # Train parameters of gaussian mixture models
    w, m, sigma = gmm_train(E_norm, gauss_pdf, n_realignment)
    
    # Estimate a posterior probability that frame isn't speech
    g0 = eval_frame_post_prob(E_norm, gauss_pdf, w, m, sigma)
    
    # Compute real VAD's markup
    vad_frame_markup_real = (g0 < vad_thr).astype('float32')  # frame VAD's markup

    vad_markup_real = np.zeros(len(signal)).astype('float32') # sample VAD's markup
    for idx in range(len(vad_frame_markup_real)):
        vad_markup_real[idx*shift:shift+idx*shift] = vad_frame_markup_real[idx]

    vad_markup_real[len(vad_frame_markup_real)*shift - len(signal):] = vad_frame_markup_real[-1]
    
    # Morphology Filters
    vad_markup_real = closing(vad_markup_real, np.ones(mask_size_morph_filt)) # close filter
    vad_markup_real = opening(vad_markup_real, np.ones(mask_size_morph_filt)) # open filter
    
    return vad_markup_real

def reverb(signal, impulse_response):
    # Function to create reverberation effect
    
    signal_reverb = np.zeros(len(signal)).astype('float32')
    
    ###########################################################
    # Here is your code
    
    ###########################################################
    
    return signal_reverb

def awgn(signal, sigma_noise):
    # Function to add white gaussian noise to signal
    
    signal_noise = np.zeros(len(signal)).astype('float32')
    
    ###########################################################
    noise = np.random.normal(0, sigma_noise, len(signal))
    signal_noise = signal + noise
    
    ###########################################################
    
    return signal_noise