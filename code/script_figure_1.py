#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 15:00:08 2021

Script using data from one subject to illustrate the process of burst calling.

Input :
raw data, information about peak frequency & channel and timing of 
spontaneous condition (i.e. which data segment to use)

Output :
one figure, Figure1_raw_data_processing, illustrating raw data, band pass 
filtered data, and amplitude envelope with bursts marked. 

"""

import numpy as np
import matplotlib.pyplot as plt
import mne

######  GENERAL ANALYSIS SETTINGS 
n_fft=2048
fmin, fmax = 2, 48.  # lower and upper band pass frequency for data filtering
lim = 100  # lower duration limit for beta burst, in ms
lim2 = 200 # lower duration limit for 'long beta burst', in ms --> used for visualization and in burst trigger figure

# Burst analaysis
# downsampling factor
dwn=3.0
percentile=75

# path information
path_trunc='../'

# paths for saving
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
        
pat_list=[5832]
        
for g in range(0, len(pat_list)):
    import glob
    txtfiles = []
    for file in glob.glob(path_trunc + 'raw_data/spontti_new_data/%s*standard.fif' % pat_list[g]):
        txtfiles.append(file)   

    
    ###### LOAD RAW DATA
    Cond=[]
    cmot=[]
    for h in range(0, len(txtfiles)): 
        fn = txtfiles[h]
        fs = fn.split("/")
        fs = fs[-1].split("_")
        pat_id=int(fs[0])
        cond=fs[2] # format different
        exptcond=fs[1]            
        print('cond %s' % cond)
        print('pat_id %s' % pat_id)
        # if cond=='pre':
        #     break
        # if cond=='ON':
        #     break
        # if cond=='OFF':
        #     c=3
        #     continue
        # if exptcond=='mot':
        #     break
        # if exptcond=='spont':
        if cond=='OFF' and exptcond=='spont':
            c=3
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
                        
        # Filtering
        # notch filtering for all channels
        picks = mne.pick_types(raw_mot.info, meg='grad', eeg=False, stim=False, eog=False, emg=True,
                                exclude='bads')
        raw_mot.filter(fmin, fmax, fir_design='firwin')
        raw_mot.notch_filter(np.arange(50, 240, 50), picks=picks, fir_design='firwin')
        
        # low pass filter for MEG channels (gradiometers)
        picks_meg = mne.pick_types(raw_mot.info, meg='grad', eeg=False, stim=False, eog=False, emg=False,
                                exclude='bads')
        raw_mot.filter(fmin, None, fir_design='firwin')
    

        # select subset of channels
        data1, times = raw_mot.get_data(picks=picks_bm, return_times=True)   
        
        # Resampling
        ### check with Jan Kujala/Jussi Nurminen what this does to the data (and whether we want to do it)
        ### 16.9.2019 - Jussi Nurminen says mne.filter.resample includes filters to avoid aliasing 
        sfreq=raw0.info['sfreq']
        down=dwn*(sfreq/1000)    
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

        import scipy as scipy
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
                    burst_dur=np.concatenate((burst_dur,tmp), axis=None)
                    burst_onset=np.concatenate((burst_onset,tmp1), axis=None)
                    burst_offset=np.concatenate((burst_offset, tmp2), axis=None)
        burst_dur_ms=(burst_dur/sfreq)*1000


        # binarized & temporally thresholded time series (bursts > lim)
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
            
                    
        #### Plot data
        ### GENERAL PLOTTING SETTINGS

        # Font specifications
        font = {'family': 'arial',
            'color':  'black',
            'weight': 'normal',
            'size': 24,
            }

        t_from=0 # start time (in seconds)
        t_to=10 # end time (in seconds)
        start=int(t_from*sfreq)
        stop=int(t_to*sfreq)
        time=np.arange(t_from, t_to, 1/sfreq)
        time=time[0:(stop-start)]
        
        ylim=np.max(filt1[start:stop])+(np.max(filt1[start:stop]))/20
        y1=burst_binary_all[start:stop]                
        
        
        fig, axs = plt.subplots(3, 1, figsize=(12, 7))
        fig.tight_layout()
        plt.subplots_adjust(bottom=0.12, left=0.02, right=0.95, top=0.92, wspace=0.25)

        # raw signal: out1
        ax1=plt.subplot(311)
        plt.plot(time, np.squeeze(out1)[start:stop], label='raw data')
        plt.title('raw data', loc='left', fontdict=font)
        frame1 = plt.gca()
        frame1.axes.get_yaxis().set_ticks([])
        frame1.axes.get_xaxis().set_ticklabels([])
        
        # narrow band filtered signal: amplitude
        ax2=plt.subplot(312)
        plt.plot(time, amplitude[start:stop], label='band pass filtered data')
        plt.title('band pass filtered data', loc='left', fontdict=font)
        frame1 = plt.gca()
        frame1.axes.get_yaxis().set_ticks([])
        frame1.axes.get_xaxis().set_ticklabels([])
        
        # smoothed amplitude envelope: filt 1
        ax3=plt.subplot(313)
        plt.plot(time, filt1[start:stop], label='amplitude envelope')
        ax3.axhline(val, color='red', lw=2, alpha=0.5)
        ax3.fill_between(time, 0, ylim, y1 > 0,
                        facecolor='red', alpha=0.3)
        ax3.fill_between(time, val, filt1[start:stop], y1 > 0,
                        facecolor='red', alpha=0.4)
        plt.title('amplitude envelope', loc='left', fontdict=font)
        plt.xticks(fontsize=20)            
        plt.xlabel('time (s)', fontdict=font)
        frame1 = plt.gca()
        frame1.axes.get_yaxis().set_ticks([])
        
        ftitle=(path_to_figs + 'Figure1_raw_data_processing_%s_%s.png' % (pat_id, side))          
        fig.savefig(ftitle) 
