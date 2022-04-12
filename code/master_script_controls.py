#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File which doees basic burst analysis. Reads in raw MEG data and for the subjects 
specific channel of interest and beta frequency of interest, computes a beta amplitude 
envelope, thresholds it at amplitude threshold 'percentile' and outputs processed 
time series and burst characteristics for further processing & plotting. 


Parameters
----------
pat_list : list of subjects to analyse
EO_times : 
    contains information about condition timing (times of MEG data to use)
beta_info_f_ch : string (path to datafile)
    contains information about subject-specific beta maximum channel & frequency per hemisphere

Returns: 
-------
output_file which contains the following variables:
    
BURST_RAW_all - individual smoothed amplitude envelope time series
BURST_DUR_all - individual burst durations
BURST_AMP_all - individual burst amplitudes
BURST_INFO_all - information on binarized timeseries run length, which has 3 entries per subject & condition:
            burst_info[0] burst duration
            burst_info[1] burst onset
            burst_info[2] burst (=1) or no burst (=0)
BURST_BBI_all - individual burst to burst interval (onset to onset)
BURST_IBI_all - individual burst to burst interval (offset to onset)
BURST_BIN_all - individual binarized time series (no temporal thresholding done, i.e. including bursts < lim)
sfreq - sampling frequency after downsampling (= SF of binarized data & amplitude envelope data)

Each variable contains two lines per subject (two hemispheres, right hemisphere first), subject order given in pat_list
Information on hemispheres is given in file beta_info_f_ch

output_file2 which contains power spectral information for the off condition

There are two patient lists pat_list in the script. One has all 21 control subjects,
the second one has 3 outlier patients removed (for the supplementary figures)


"""

import numpy as np
import mne
import scipy as scipy
import glob
import pickle

######  GENERAL ANALYSIS SETTINGS
# other analysis parameters
n_fft=512
fmin, fmax = 2, 48.  # lower and upper band pass frequency for data filtering
lim = 100  # lower duration limit for beta burst, in ms
lim2 = 200 # lower duration limit for 'long beta burst', in ms
percentile=75

# Burst analysis
# downsampling target frequency (Hz)
dwn=200

# path information
path_trunc='../'
path_to_figs=path_trunc +'figures/'
path_to_rawdat=path_trunc+ 'raw_data/'
path_to_procdat=path_trunc+ 'processed_data/'

###### INFORMATION ON RAW DATA INTERVALS TO USE
timing_info=path_to_rawdat + 'MEG_times_controls.pkl' 
with open(timing_info, 'rb') as file:  # Python 3: open(..., 'rb')
      EO_times = pickle.load(file)
      
# load information about peak beta channels & peak beta frequencies
import info_beta_freqs_chans
beta_info_f_ch=info_beta_freqs_chans.info_beta_freqs_chans()
    
# all controls
pat_list=['sib24', 'sib26', 'sib85', 'sib124', 'sib155', 'sib212','s09', 's10', 's11', 's15',  's16', 's17', 's18', 's19', 's20', 's21', 's22', 's23', 's24',  's25', 's26']
output_file = 'controls_all.pkl'
output_file2 = 'controls_all_psd_spectra.pkl'

# controls without outliers 
# (sib26, sib124 and s19 removed b/c of repeatedly being outliers in various domains)
# pat_list=['sib24', 'sib85', 'sib155', 'sib212','s09', 's10', 's11', 's15',  's16', 's17', 's18', 's20', 's21', 's22', 's23', 's24',  's25', 's26']
# output_file = 'controls_outliers_removed.pkl'

BURST_RAW_all=[]
BURST_DUR_all=[]
BURST_AMP_all=[]
BURST_INFO_all=[]
BURST_BIN_all=[]
BURST_BBI_all=[]
BURST_IBI_all=[]
PSD=list()
FREQUENCIES=list()
for g in range(0, len(pat_list)):
      
    txtfiles = []
    for file in glob.glob(path_trunc + 'raw_data/controls/%s*sss.fif' % pat_list[g]):
        txtfiles.append(file)   

    ###### LOAD RAW DATA
    fn = txtfiles[0]
    fs = fn.split("/")
    fs = fs[-1].split("_")
    pat_id=fs[0]
    print('pat_id %s' % pat_id)

    for i in range(0, len(EO_times)):
        if EO_times[i][0]==pat_id:
            info_t=EO_times[i]
            print(info_t)
        
    # load raw data: eyes open (info_t[2]) condition
    tmin=info_t[1]
    tmax=info_t[2]
    raw=mne.io.read_raw_fif(fn).crop(tmin, tmax).load_data()

            
    
    ################################################################################################################
    ################################################################################################################
    #################### COMPUTE POWER SPECTRAL DENSITY (PSD) & TOPOPLOTS ##########################################
    ################################################################################################################
    ################################################################################################################
    
    # ###### COMPUTE POWER SPECTRAL DENSITY (PSD)
    modality='topoplots'
    sfreq=raw.info['sfreq']
        
    # pick subset of channels (gradiometers)
    picks_meg = mne.pick_types(raw.info, meg='grad', eeg=False, stim=False, eog=False, emg=False, exclude='bads')
    ch_names = np.asarray(raw.ch_names)
    channels=ch_names[picks_meg]
    raw_chans=raw.pick_channels(channels)
    
    # filtering (fmin, fmax defined earlier)
    raw_chans.filter(fmin, fmax, fir_design='firwin')
    
    # pick nfft according to sfreq
    if sfreq==600:
        n_fft=512
    if sfreq==1000:
        n_fft=1024
    print('nfft= %s' % n_fft)
    
    # calculate PSD
    # psds, freqs = mne.time_frequency.psd_welch(raw_chans, tmin=None, tmax=None,
    #                         fmin=fmin, fmax=fmax, n_fft=n_fft)
    
    # ################################################################################################################
    # ######################################### PLOTTING ROUTINE #####################################################
    # ################################################################################################################
    
    # fmin_plot=12
    # fmax_plot=30
    # b_idx=np.where(np.logical_and(freqs>fmin_plot, freqs<fmax_plot))

    # def my_callback(ax, ch_idx):
    #     """
    #     This block of code is executed once you click on one of the channel axes
    #     in the plot. To work with the viz internals, this function should only take
    #     two parameters, the axis and the channel or data index.
    #     """
    #     ax.plot(freqs, psds[ch_idx], color='red')
    #     ax.set_xlabel = 'Frequency (Hz)'
    #     ax.set_ylabel = 'Power (dB)'
    
    
    # for ax, idx in mne.viz.iter_topography(raw_chans.info,
    #                                 fig_facecolor='white',
    #                                 axis_facecolor='white',
    #                                 axis_spinecolor='white',
    #                                 on_pick=my_callback):
    #     ax.plot(psds[idx][b_idx], color='red')
        
    
    # plt.gcf().suptitle('PSD, visible range %s-%s Hz, %s' % (fmin_plot, fmax_plot, pat_id))
    # plt.gcf().set_size_inches(12, 12) 
    
    # ftitle=('/PSD_topo_ctrl_%s-%s_%s.pdf' % (fmin_plot, fmax_plot, pat_id)) 
    # fname  = path_to_figs + modality + ftitle
    # plt.gcf().savefig(fname)
    # plt.show() 
    
    
    # picks_lm=['MEG0222','MEG0223','MEG0413','MEG0412','MEG0422','MEG0423']
    # picks_rm=['MEG1112','MEG1113','MEG1123','MEG1122','MEG1312','MEG1313']
    
    # raw_lm=raw.copy().pick_channels(picks_lm)
    # raw_rm=raw.copy().pick_channels(picks_rm)
    
    # psds_lm, freqs = mne.time_frequency.psd_welch(raw_lm, tmin=None, tmax=None,
    #                         fmin=fmin, fmax=fmax, n_fft=n_fft)

    
    # psds_rm, freqs = mne.time_frequency.psd_welch(raw_rm, tmin=None, tmax=None,
    #                         fmin=fmin, fmax=fmax, n_fft=n_fft)
    
    
    # # plot channels on top of each other per side, for finding top channel & frequency
    
    # fig, axs = plt.subplots(1, 2, figsize=(8,10))
    
    # plt.subplot(121)
    # for idx in range(0,len(psds_lm)):
    #     plt.plot(freqs[b_idx], psds_lm[idx][b_idx], label=picks_lm[idx])
    # plt.title('left')
    # plt.legend()
    
    # plt.subplot(122)
    # for idx in range(0,len(psds_rm)):
    #     plt.plot(freqs[b_idx], psds_rm[idx][b_idx], label=picks_rm[idx])
    # plt.title('right')
    # plt.legend()
    #%%
    ################################################################################################################
    ################################################################################################################
    ################################ BETA BURST ANALYSIS ###########################################################
    ################################################################################################################
    ################################################################################################################
    
    
    ###### MEG CHANNEL SELECTION AND BETA PEAK FREQUENCIES FOR FURTHER ANALYSIS
    # channel selection
    Beta_lo=[]
    Beta_hi=[]
    Picks_bm=[]
    for j in range(0, len(beta_info_f_ch)):
        if beta_info_f_ch[j][0]==pat_id:
            print('patient found')
            tmp1=beta_info_f_ch[j][1]
            tmp2=beta_info_f_ch[j][2]
            tmp3=beta_info_f_ch[j][3]
            Beta_lo.append(tmp1)
            Beta_hi.append(tmp2)
            Picks_bm.append(tmp3)
            tmp1=beta_info_f_ch[j][4]
            tmp2=beta_info_f_ch[j][5]
            tmp3=beta_info_f_ch[j][6]
            Beta_lo.append(tmp1)
            Beta_hi.append(tmp2)
            Picks_bm.append(tmp3)  
            
    Psd=list()
    Frequencies=list()
    for idx in range(0, len(Picks_bm)): # first right, then left hemisphere
        picks_bm=Picks_bm[idx]
        if picks_bm ==[]:
            continue
        beta_lo=Beta_lo[idx]
        beta_hi=Beta_hi[idx]
        if idx==0:
            side='left'
        if idx==1:
            side='right'
            

        ######  BETA BURST ANALYSIS AND PLOTTING OF BASIC BURST METRICS 
        # Filtering
        # notch filtering for all channels
        picks = mne.pick_types(raw.info, meg='grad', eeg=False, stim=False, eog=False, emg=True,
                                exclude='bads')
        raw.notch_filter(np.arange(50, 240, 50), picks=picks, fir_design='firwin')
        
        # low pass filter for MEG channels (gradiometers)
        picks_meg = mne.pick_types(raw.info, meg='grad', eeg=False, stim=False, eog=False, emg=False,
                                exclude='bads')
        raw.filter(fmin, None, picks=picks_meg, fir_design='firwin')
        
        # calculate PSD
        psd, freqs = mne.time_frequency.psd_welch(raw, picks=picks_bm, tmin=None, tmax=None,
                                fmin=fmin, fmax=fmax, n_fft=n_fft)
        Psd.append(psd)
        Frequencies.append(freqs)
        
        # select subset of channels
        data1, times = raw.get_data(picks=picks_bm, return_times=True)   
        
        # Resampling
        ###  mne.filter.resample includes filters to avoid aliasing 
        sfreq=raw.info['sfreq']
        down=sfreq/dwn   
        # goal: downsampling to frequency of 333 Hz, i.e. multiply by factor sfreq/1000 
        out1= mne.filter.resample(data1, down=down, npad='auto', n_jobs=16, pad='reflect_limited', verbose=None) # additional options: window='boxcar', npad=100,
    
        # split data into consecutive epochs
        sfreq=raw.info['sfreq']/down
        ws=int(20*sfreq/fmin) # number of samples per window
        overlap=1-0 # set amount of overlap for consecutive FFT windows (second number sets amount of overlap)
    
        # separate data into consecutive data chunks (episode-like, because spectral_connectivity expects epochs)
        array1 = list()
        start = 0
        stop=ws
        step = int(ws*overlap)
        while stop < out1.shape[1]:
            tmp = out1[:, start:stop]
            start += step
            stop += step
            array1.append(tmp)
    
        # define frequencies of interest
        freqs = np.arange(7., 47., 1.)
        n_cycles = freqs / 2.
        #n_cycles = np.arange(2,15,1)
        
        power = mne.time_frequency.tfr_array_morlet(array1, sfreq=sfreq, freqs=freqs,
                            n_cycles=n_cycles, output='complex', n_jobs=16)
    
        freq_lo=beta_lo-int(min(freqs)) 
        freq_hi=beta_hi-int(min(freqs))
        cutoff=np.ceil(lim/1000*sfreq) ### --> multiply with sfreq to get value in data points


        amplitude=[]
        for k in range(0,len(power)):
            tmp=power[k][0][freq_lo:freq_hi+1]
            tmptmp=np.mean(tmp, axis=0)
            amplitude=np.concatenate((amplitude,tmptmp), axis=None)
        rec_amp=np.absolute(amplitude)# , /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])
        
        fwhm=sfreq/(5*down)
        def fwhm2sigma(fwhm):
            return fwhm / np.sqrt(8 * np.log(2))
        sigma = fwhm2sigma(fwhm)

        filt1=scipy.ndimage.filters.gaussian_filter1d(rec_amp, sigma, axis=-1, order=0, mode='reflect', cval=0.0, truncate=4.0)
        val=np.percentile(filt1, percentile)  

        bin_burst =(filt1 > val).astype(np.int_) # outputs binarized data, 1 for values above threshold

        ### copied from stackoverflow
        ### https://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi
        def rle(inarray):
            ## run length encoding. Partial credit to R rle function. 
            ## Multi datatype arrays catered for including non Numpy
            ## returns: tuple (runlengths, startpositions, values) """
            ia = np.asarray(inarray)                  # force numpy
            n = len(ia)
            if n == 0: 
                return (None, None, None)
            else:
                y = np.array(ia[1:] != ia[:-1])     # pairwise unequal (string safe)
                i = np.append(np.where(y), n - 1)   # must include last element posi
                z = np.diff(np.append(-1, i))       # run lengths
                p = np.cumsum(np.append(0, z))[:-1] # positions
                return(z, p, ia[i]) # return(z, p, ia[i])
        
        burst_dur=[]
        burst_dur_ms=[]
        burst_amp=[]
        burst_onset=[]
        burst_offset=[]
        
        burst_info = rle(bin_burst) 
        for l in range(0,len(burst_info[0])):
            if burst_info[2][l]>0:
                if burst_info[0][l]>=cutoff:
                    tmp=burst_info[0][l]    # burst duration
                    tmp1=burst_info[1][l]   # burst onset
                    tmp2=tmp1+tmp           # burst offset
                    tmp3=np.max(filt1[tmp1:tmp1+tmp]) # burst amplitude
                    burst_dur=np.concatenate((burst_dur,tmp), axis=None)
                    burst_onset=np.concatenate((burst_onset,tmp1), axis=None)
                    burst_amp=np.concatenate((burst_amp,tmp3), axis=None)
                    burst_offset=np.concatenate((burst_offset, tmp2), axis=None)
        burst_dur_ms=(burst_dur/sfreq)*1000
        
        bbi=(np.diff(burst_onset))/sfreq*1000

        
        ibi=[]
        for l in range(1,len(burst_offset)):
            tmp=burst_onset[l]-burst_offset[l-1]
            ibi=np.concatenate((ibi,tmp), axis=None)
        ibi=(ibi/sfreq)*1000

            
        BURST_DUR_all.append(burst_dur_ms)
        BURST_AMP_all.append(burst_amp)
        BURST_INFO_all.append(burst_info)
        BURST_RAW_all.append(filt1)  ### raw amplitude data (filtered)
        BURST_BIN_all.append(bin_burst)
        BURST_BBI_all.append(bbi)
        BURST_IBI_all.append(ibi)
    PSD.append(Psd)
    FREQUENCIES.append(Frequencies)            
        
###############################################################################################################
###############################################################################################################
############################### SAVE VARIABLES FOR LATER GROUP STATISTICS #####################################
###############################################################################################################
###############################################################################################################


import pickle

# Saving the objects (bursting info)
with open(path_to_procdat + output_file, 'wb') as file:
    pickle.dump([BURST_RAW_all, BURST_DUR_all, BURST_AMP_all, BURST_INFO_all, BURST_BBI_all, BURST_IBI_all, BURST_BIN_all, sfreq, pat_list], file)

# saving PSD information
with open(path_to_procdat + output_file2, 'wb') as file:
    pickle.dump([PSD, FREQUENCIES, pat_list], file)
