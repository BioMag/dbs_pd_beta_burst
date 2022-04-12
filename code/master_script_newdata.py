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

idx_cond_trunc : contains information about condition timing (times of MEG data to use)
Format (example entry): 
[array([4.000e+00, 5.874e+03, 3.000e+00, 2.000e+00]),     #### Pat number, patient id, condition (2=ON, 3=OFF), hand used (1=left)
 array([320., 495.]),                                     #### EC condition, beginning and end
 array([ 10., 305., 515., 815.]),                         #### EO condition, beginning and end (pairs)
 array([ 12.,  55., 108., 150., 186., 228., 289., 320.])] #### MOT condition, beginning and end (pairs) - this data is also in a different data file

beta_info_f_ch : string (path to datafile)
    contains information about subject-specific beta maximum channel & frequency

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

Each variable contains two entries per subject (two hemispheres, right hemisphere first), subject order given in pat_list
Each entry has two entries (DBS OFF and ON condition, OFF first)
Information on hemispheres is given in file beta_info_f_ch

output_file2 which contains power spectral information for the off condition


"""
import numpy as np
import mne
import scipy as scipy
import glob
import pickle

######  GENERAL ANALYSIS SETTINGS 
n_fft=2048
fmin, fmax = 2, 48.  # lower and upper band pass frequency for data filtering
lim = 100  # lower duration limit for beta burst, in ms
lim2 = 200 # lower duration limit for 'long beta burst', in ms

# Burst analysis
# downsampling frequency target (Hz)
dwn=200
percentile=75


# Path for saving data

# path information
path_trunc='../'

# paths for saving
path_to_figs=path_trunc +'figures/'
path_to_rawdat=path_trunc+ 'raw_data/'
path_to_procdat=path_trunc+ 'processed_data/'

###### INFORMATION ON RAW DATA INTERVALS TO USE
timing_info=path_to_rawdat + 'MEG_times_new_patients.pkl' 
with open(timing_info, 'rb') as file:  # Python 3: open(..., 'rb')
      idx_cond_trunc = pickle.load(file)

# load information about peak beta channels & peak beta frequencies
import info_beta_freqs_chans
beta_info_f_ch=info_beta_freqs_chans.info_beta_freqs_chans()

pat_list=[5815, 5826, 5832, 5874, 5875, 5912, 5926, 5938, 5952, 5953, 5962, 5976, 5987]
output_file = 'new_patients.pkl'
output_file2 = 'new_patients_psd_spectra.pkl'

# excluded patients: 
# 5923: no clear beta peak, patient was constantly falling asleep during measurement

###### VARIABLE NAMES FOR LATER DATA SAVING
BURST_RAW_all=[]
BURST_DUR_all=[]
BURST_AMP_all=[]
BURST_INFO_all=[]
BURST_BBI_all=[]
BURST_IBI_all=[]
BURST_BIN_all=[]
PSD=list()
FREQUENCIES=list()
for g in range(0, len(pat_list)):
    txtfiles = []
    for file in glob.glob(path_trunc + 'raw_data/spontti_new_data/%s*standard.fif' % pat_list[g]):
        txtfiles.append(file)   

    
    ###### LOAD RAW DATA
    raw_MOT=[]
    Cond=[]
    cmot=[]
    for h in range(0, len(txtfiles)): 
        fn = txtfiles[h]
        fs = fn.split("/")
        fs = fs[-1].split("_")
        pat_id=int(fs[0])
        cond=fs[2]
        exptcond=fs[1]

        print('cond %s' % cond)
        print('pat_id %s' % pat_id)
        if cond=='ON':
            c=2
        if cond=='OFF':
            c=3
        if exptcond=='spont':
            for i in range(0, len(idx_cond_trunc)):
                if np.logical_and(int(idx_cond_trunc[i][0][1])==pat_id, int(idx_cond_trunc[i][0][2])==c):
                    info_t=idx_cond_trunc[i]
                    
            # load raw data: eyes closed (info_t[1]) or eyes open (info_t[2]) condition
            raws=list()
            raw_mot=[]
            Cond.append(cond)
            for l in range(1, int(len(info_t[2])/2)+1):
                tmin=info_t[2][l*2-2] # use analysis times from idx_cond
                tmax=info_t[2][l*2-1] # use analysis times from idx_cond
                raw0=mne.io.read_raw_fif(fn).crop(tmin, tmax).load_data()
                raws.append(raw0)
            raw_mot=mne.concatenate_raws(raws)
            raw_mot.info['bads']
            raw_MOT.append(raw_mot)
            cmot.append(c)
        
    # check & standardize condition order (OFF first)
    print('Condition order check: %s condition first' % Cond[0])
    if Cond[0]=='ON':
        tmp1=Cond[0]
        tmp2=Cond[1]
        Cond=[tmp2, tmp1]
        tmp1=raw_MOT[0]
        tmp2=raw_MOT[1]
        raw_MOT=[tmp2, tmp1]
    print('Condition order after check: %s condition first' % Cond[0])
    
    
    
    ################################################################################################################
    ################################################################################################################
    #################### COMPUTE POWER SPECTRAL DENSITY (PSD) & TOPOPLOTS ##########################################
    ################################################################################################################
    ################################################################################################################
    
    # ###### COMPUTE POWER SPECTRAL DENSITY (PSD)
    # modality='topoplots'
    # sfreq=raw0.info['sfreq']
    # PSDS=[]
    # for i in range(0, len(raw_MOT)):
        
    #     # pick subset of channels (gradiometers)
    #     picks_meg = mne.pick_types(raw_MOT[i].info, meg='grad', eeg=False, stim=False, eog=False, emg=False, exclude='bads')
    #     ch_names = np.asarray(raw_MOT[i].ch_names)
    #     channels=ch_names[picks_meg]
    #     raw_chans=raw_MOT[i].pick_channels(channels)
        
    #     # filtering (fmin, fmax defined earlier)
    #     raw_chans.filter(fmin, fmax, fir_design='firwin')
        
    #     # calculate PSD
    #     psds, freqs = mne.time_frequency.psd_welch(raw_chans, tmin=None, tmax=None,
    #                             fmin=fmin, fmax=fmax, n_fft=n_fft)
        
    #     PSDS.append(psds)
        
    
    # # ################################################################################################################
    # # ######################################### PLOTTING ROUTINE #####################################################
    # # ################################################################################################################
    
    # fmin_plot=12
    # fmax_plot=30
    # b_idx=np.where(np.logical_and(freqs>fmin_plot, freqs<fmax_plot))
    
    # psds1=PSDS[0]
    # psds2=PSDS[1]
    
    # def my_callback(ax, ch_idx):
    #     """
    #     This block of code is executed once you click on one of the channel axes
    #     in the plot. To work with the viz internals, this function should only take
    #     two parameters, the axis and the channel or data index.
    #     """
    #     ax.plot(freqs, psds1[ch_idx], color='red')
    #     ax.plot(freqs, psds2[ch_idx], color='blue')
    #     ax.set_xlabel = 'Frequency (Hz)'
    #     ax.set_ylabel = 'Power (dB)'
    
    
    # for ax, idx in mne.viz.iter_topography(raw_chans.info,
    #                                 fig_facecolor='white',
    #                                 axis_facecolor='white',
    #                                 axis_spinecolor='white',
    #                                 on_pick=my_callback):
    #     ax.plot(psds1[idx][b_idx], color='red')
    #     ax.plot(psds2[idx][b_idx], color='blue')
        
    
    # plt.gcf().suptitle('PSD, OFF (red) vs ON (blue), visible range %s-%s Hz, %s' % (fmin_plot, fmax_plot, pat_id))
    # plt.gcf().set_size_inches(12, 12) 
    
    # ftitle=('/PSD_topo_ON_vs_OFF_%s-%s_%s.pdf' % (fmin_plot, fmax_plot, pat_id)) 
    # fname  = path_to_figs + modality + ftitle
    # plt.gcf().savefig(fname)
    # plt.show() 
    
    
    ################################################################################################################
    ################################################################################################################
    ################################ BETA BURST ANALYSIS ###########################################################
    ################################################################################################################
    ################################################################################################################
        
    # channel selection
    Beta_lo=[]
    Beta_hi=[]
    Picks_bm=[]
    for j in range(0, len(beta_info_f_ch)):
        if beta_info_f_ch[j][0]==int(pat_id):
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
        BURST_thresh=[]
        BURST_raw=[]
        BURST_info=[]
        BURST_dur_ms=[]   
        BURST_amp=[]
        BURST_start=[]
        BIN_burst=[]
        BBI=[]
        IBI=[]
        for i in range(0, len(raw_MOT)):
                        
            # Filtering
            # notch filtering for all channels
            picks = mne.pick_types(raw_MOT[i].info, meg='grad', eeg=False, stim=False, eog=False, emg=True,
                                    exclude='bads')
            raw_MOT[i].filter(fmin, fmax, fir_design='firwin')
            raw_MOT[i].notch_filter(np.arange(50, 240, 50), picks=picks, fir_design='firwin')
            
            # low pass filter for MEG channels (gradiometers)
            picks_meg = mne.pick_types(raw_MOT[i].info, meg='grad', eeg=False, stim=False, eog=False, emg=False,
                                    exclude='bads')
            raw_MOT[i].filter(fmin, None, fir_design='firwin')
        
            ###### COMPUTE POWER SPECTRAL DENSITY (PSD)
            sfreq=raw0.info['sfreq']
                
            # pick subset of channels (gradiometers)
            picks_meg = mne.pick_types(raw_MOT[i].info, meg='grad', eeg=False, stim=False, eog=False, emg=False, exclude='bads')
            ch_names = np.asarray(raw_MOT[i].ch_names)
            channels=ch_names[picks_meg]
            raw_chans=raw_MOT[i].pick_channels(channels)
            
            # filtering (fmin, fmax defined earlier)
            raw_chans.filter(fmin, fmax, fir_design='firwin')
            
            # get OFF condition PSD curves
            if i==0:
                # calculate PSD
                psd, freqs = mne.time_frequency.psd_welch(raw_MOT[i], picks=picks_bm, tmin=None, tmax=None,
                                        fmin=fmin, fmax=fmax, n_fft=n_fft)
                Psd.append(psd)
                Frequencies.append(freqs)

            # select subset of channels
            data1, times = raw_MOT[i].get_data(picks=picks_bm, return_times=True)   
            
            # Resampling
            ###  mne.filter.resample includes filters to avoid aliasing 
            sfreq=raw0.info['sfreq']
            down=sfreq/dwn   
            # downsampling to frequency of 333 Hz
            out1= mne.filter.resample(data1, down=down, npad='auto', n_jobs=16, pad='reflect_limited', verbose=None) # additional options: window='boxcar', npad=100,
        
            # split data into consecutive epochs
            sfreq=raw0.info['sfreq']/down
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
            burst_onset=[]
            burst_dur_ms=[]
            burst_amp=[]
            burst_offset=[]
            cutoff=np.ceil(lim/1000*sfreq) ### --> multiply with sfreq to get value in data points

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
            
            BURST_raw.append(filt1)  ### raw amplitude data (filtered)                
            BURST_info.append(burst_info)
            BURST_dur_ms.append(burst_dur_ms)
            BURST_amp.append(burst_amp)
            BIN_burst.append(bin_burst)
            BBI.append(bbi)
            IBI.append(ibi)
            
        BURST_RAW_all.append(BURST_raw)
        BURST_DUR_all.append(BURST_dur_ms)
        BURST_AMP_all.append(BURST_amp)
        BURST_INFO_all.append(BURST_info)
        BURST_BBI_all.append(BBI)
        BURST_IBI_all.append(IBI)
        BURST_BIN_all.append(BIN_burst)
    PSD.append(Psd)
    FREQUENCIES.append(Frequencies)    

    
###############################################################################################################
###############################################################################################################
############################### SAVE VARIABLES FOR LATER GROUP STATISTICS #####################################
###############################################################################################################
###############################################################################################################


import pickle

# Saving the objects: burst information
with open(path_to_procdat + output_file, 'wb') as file:
    pickle.dump([BURST_RAW_all, BURST_DUR_all, BURST_AMP_all, BURST_INFO_all, BURST_BBI_all, BURST_IBI_all, BURST_BIN_all, sfreq, pat_list], file)
        # Saving the objects
        
# saving PSD information
with open(path_to_procdat + output_file2, 'wb') as file:
    pickle.dump([PSD, FREQUENCIES, pat_list], file)
