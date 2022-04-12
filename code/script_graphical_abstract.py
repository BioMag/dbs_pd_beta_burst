#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 15:00:08 2021

@author: amande
"""

import numpy as np
import matplotlib.pyplot as plt
import mne
import glob
import scipy as scipy

######  GENERAL ANALYSIS SETTINGS
fmin, fmax = 2, 48.  # lower and upper band pass frequency for data filtering
lim = 100  # lower duration limit for beta burst, in ms

# Burst analysis
# downsampling frequency target (Hz)
dwn = 200
percentile=75

# path information
path_trunc='../'
path_to_figs=path_trunc +'figures/'
path_to_procdat=path_trunc+ 'processed_data/'

###### INFORMATION ON RAW DATA INTERVALS TO USE
import timing_info_newdata

## extract useable MEG segments for each of the condition times
cond_info=path_trunc + 'raw_data/spontti_new_data/pd_condition_times'
MEG_info=path_trunc + 'raw_data/spontti_new_data/pd_meg_times'

idx_cond_trunc = timing_info_newdata.timing_info_newdata(cond_info, MEG_info)

# load information about peak beta channels & peak beta frequencies
import info_beta_freqs_chans
beta_info_f_ch=info_beta_freqs_chans.info_beta_freqs_chans()
        
pat_list=[5976]

BURST_binary=list()
VAL=list()
FILT=list()
for g in range(0, len(pat_list)):
    txtfiles = []
    for file in glob.glob(path_trunc + 'raw_data/spontti_new_data/%s*standard.fif' % pat_list[g]):
        txtfiles.append(file)   

    ###### LOAD RAW DATA
    raw_MOT=[]
    Cond=[]
    for h in range(0, len(txtfiles)): 
        fn = txtfiles[h]
        fid=fn[29:]        
        fs = fid.split("_")
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
        
            # select subset of channels
            data1, times = raw_MOT[i].get_data(picks=picks_bm, return_times=True)   
            
            # Resampling
            ### check with Jan Kujala/Jussi Nurminen what this does to the data (and whether we want to do it)
            ### 16.9.2019 - Jussi Nurminen says mne.filter.resample includes filters to avoid aliasing 
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
            FILT.append(filt1)
            VAL.append(val)            
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
            
            cutoff=np.ceil(lim/1000*sfreq) ### --> multiply with sfreq to get value in data points
            burst_info = rle(bin_burst) 
            zeros = [0] * len(bin_burst)
            
            # binarized timeseries, all bursts
            burst_binary_all=[]
            for k in range(0, len(burst_info[0])):
                tmp4=[]
                bdur=burst_info[0][k]
                bonset=burst_info[1][k]
                if burst_info[2][k]>0:
                    if burst_info[0][k]>=cutoff:
                        tmp2=zeros[bonset:bonset+bdur] 
                        for x in tmp2:
                            tmp4.append(1)   # binarized & duration thresholded trace                            
                    else:
                        tmp4=zeros[bonset:bonset+bdur] 
                else:
                    tmp4=zeros[bonset:bonset+bdur]   
                burst_binary_all=np.concatenate((burst_binary_all,tmp4), axis=None)
            BURST_binary.append(burst_binary_all)
            
                            
###### INFORMATION ON RAW DATA INTERVALS TO USE
import timing_info_control
EO_times = timing_info_control.timing_info_control()
    
pat_list=[ 's10']

for g in range(0, len(pat_list)):
    txtfiles = []
    #for file in glob.glob(path_trunc + 'raw_data/controls/*sss.fif' % pat_list[g]):
    for file in glob.glob(path_trunc + 'raw_data/controls/%s*sss.fif' % pat_list[g]):
        txtfiles.append(file)   

    ###### LOAD RAW DATA
    fn = txtfiles[0]
    fid=fn[21:]
    fs = fid.split("_")
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
    
    for idx in range(0, 1): #len(Picks_bm)): # first right, then left hemisphere
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
    
        # select subset of channels
        data1, times = raw.get_data(picks=picks_bm, return_times=True)   
        
        # Resampling
        ### mne.filter.resample includes filters to avoid aliasing 
        sfreq=raw.info['sfreq']
        down=sfreq/dwn  
        # goal: downsampling to frequency of 333 Hz, i.e. multiply by factor sfreq/1000 
        out1= mne.filter.resample(data1, down=down, npad='auto', n_jobs=16, pad='reflect_limited', verbose=None) # additional options: window='boxcar', npad=100,
    
        # split data into consecutive epochs
        sfreq=dwn
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
        FILT.append(filt1)
        VAL.append(val)
        
        bin_burst =(filt1 > val).astype(np.int_) # outputs binarized data, 1 for values above threshold
        burst_info = rle(bin_burst)     
        zeros = [0] * len(bin_burst)
            
        # binarized timeseries, all bursts
        burst_binary_all=[]
        for k in range(0, len(burst_info[0])):
            tmp4=[]
            bdur=burst_info[0][k]
            bonset=burst_info[1][k]
            if burst_info[2][k]>0:
                if burst_info[0][k]>=cutoff:
                    tmp2=zeros[bonset:bonset+bdur] 
                    for x in tmp2:
                        tmp4.append(1)   # binarized & duration thresholded trace                            
                else:
                    tmp4=zeros[bonset:bonset+bdur] 
            else:
                tmp4=zeros[bonset:bonset+bdur]   
            burst_binary_all=np.concatenate((burst_binary_all,tmp4), axis=None)
        BURST_binary.append(burst_binary_all)
        
#### Plot data
### GENERAL PLOTTING SETTINGS

# Font specifications
font = {'family': 'arial',
    'color':  'black',
    'weight': 'normal',
    'size': 24,
    }

t_from=0 # start time (in seconds)
t_to=15 # end time (in seconds)
start=int(t_from*sfreq)
stop=int(t_to*sfreq)
time=np.arange(t_from, t_to, 1/sfreq)
time=time[0:(stop-start)]

fig, axs = plt.subplots(3, 1, figsize=(12, 7))
fig.tight_layout()
plt.subplots_adjust(bottom=0.12, left=0.02, right=0.95, top=0.92, wspace=0.25)

ylim1=np.max(FILT[0][start:stop])+(np.max(FILT[0][start:stop]))/20
ylim2=np.max(FILT[1][start:stop])+(np.max(FILT[1][start:stop]))/20
ylim3=np.max(FILT[2][start:stop])+(np.max(FILT[2][start:stop]))/20
ylim=np.max([ylim1, ylim2, ylim3])

# smoothed amplitude envelope: filt 1
ax3=plt.subplot(311)
burst_binary_all=BURST_binary[0]
filt1=FILT[0]
val=VAL[0]
y1=burst_binary_all[start:stop]                
plt.plot(time, filt1[start:stop], label='amplitude envelope')
ax3.axhline(val, color='red', lw=2, alpha=0.5)
ax3.fill_between(time, 0, ylim, y1 > 0,
                facecolor='red', alpha=0.1)
ax3.fill_between(time, val, filt1[start:stop], y1 > 0,
                facecolor='red', alpha=0.4)
plt.xticks(fontsize=20)   
plt.xlim(t_from-0.5,t_to+0.5)         
plt.ylabel('DBS OFF', fontdict=font)

ax3=plt.subplot(312)
burst_binary_all=BURST_binary[1]
filt1=FILT[1]
val=VAL[1]
y1=burst_binary_all[start:stop]                
plt.plot(time, filt1[start:stop], label='amplitude envelope')
ax3.axhline(val, color='red', lw=2, alpha=0.5)
ax3.fill_between(time, 0, ylim, y1 > 0,
                facecolor='red', alpha=0.1)
ax3.fill_between(time, val, filt1[start:stop], y1 > 0,
                facecolor='red', alpha=0.4)
plt.xticks(fontsize=20)           
plt.xlim(t_from-0.5,t_to+0.5)         
plt.ylabel('DBS ON', fontdict=font)

ax3=plt.subplot(313)
burst_binary_all=BURST_binary[2]
filt1=FILT[2]
val=VAL[2]
t_from1=15 # start time (in seconds)
t_to1=30 # end time (in seconds)
start=int(t_from1*sfreq)
stop=int(t_to1*sfreq)
time=np.arange(t_from1, t_to1, 1/sfreq)
time=time[0:(stop-start)]
y1=burst_binary_all[start:stop]                
plt.plot(time, filt1[start:stop], label='amplitude envelope')
ax3.axhline(val, color='red', lw=2, alpha=0.5)
ax3.fill_between(time, 0, ylim, y1 > 0,
                facecolor='red', alpha=0.1)
ax3.fill_between(time, val, filt1[start:stop], y1 > 0,
                facecolor='red', alpha=0.4)
plt.xticks(fontsize=20)          
plt.xlim(t_from1-0.5,t_to1+0.5)         
plt.xlabel('time (s)', fontdict=font)
plt.ylabel('CONTROL', fontdict=font)

ftitle=(path_to_figs + 'graphical_abstract_raw_data_illustration.png')          
fig.savefig(ftitle) 