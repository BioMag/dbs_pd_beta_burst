#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The file uses the output from basic burst analysis script to analyse the time 
series' bursting behavior following a burst. The analysis is done on a binarized 
time series (0 = no burst, 1 = burst). 
First, all bursts are investigated. Second, bursts are separated into short 
(> lim, < lim2) and long (> lim2) bursts, and time series behaviour following 
these is examined separately. 
The time window analysed following bursts is given by parameter 'window'. 
Offset of bursts is set to be time 0. Mean value over the whole binarized time 
series is used as baseline bursting probability.


Parameters
----------
INPUT PARAMETERS USED:
    BURST_INFO_all - information on binarized timeseries characteristics
    dispersion_coefficient.pkl - data file produced by script script_figures_2_to_4

The file works on the output_file data produces by the basic burst analysis scripts
for all three subject groups, 
(1) controls, 
(2) old and 
(3) new PD DBS subjects
    
The output_file (pickle file containing data variables) contains the following variables:
    
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

Returns: 
-------

2 figures (Figure_5 and Figure_6) and stats related to re-burst behaviour

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy as scipy
from scipy.stats import sem

######  GENERAL ANALYSIS SETTINGS
lim = 100  # lower duration limit for beta burst, in ms
percentile=75
lim2 = 200 # lower duration limit for 'long beta burst', in ms

# length of time window to look at
window=30   # time in seconds

# smoothing factor for later smoothing
smoothing = 1 # 1=1 s, 0.5 = 2 s, 2=0.5 s kernel width

# path information
path_trunc='../'
path_to_figs=path_trunc +'figures/'
path_to_procdat=path_trunc+ 'processed_data/' 

import pickle

##### LOAD CONTROL DATA
with open(path_trunc + 'processed_data/controls_outliers_removed.pkl', 'rb') as file:  # Python 3: open(..., 'rb')
     BURST_RAW_all_c, BURST_DUR_all_c, BURST_AMP_all_c, BURST_INFO_all_c, BURST_BBI_all_c, BURST_IBI_all_c, BURST_BIN_all_c, sfreq_c, pat_list = pickle.load(file)

##### LOAD PATIENT DATA
BI_all=[]
BB_all=[]

##### OLD DATA FIRST, n=3 with stable medication
with open(path_to_procdat + 'old_patients.pkl' , 'rb') as file:  # Python 3: open(..., 'rb')
     BURST_RAW_all, BURST_DUR_all, BURST_AMP_all, BURST_INFO_all, BURST_BBI_all, BURST_IBI_all, BURST_BIN_all, sfreq_p, pat_list = pickle.load(file)
  
BI_all.append(BURST_INFO_all)
BB_all.append(BURST_BIN_all)

###### NEW DATA NEXT
with open(path_to_procdat + 'new_patients.pkl', 'rb') as file:  # Python 3: open(..., 'rb')
     BURST_RAW_all, BURST_DUR_all, BURST_AMP_all, BURST_INFO_all, BURST_BBI_all, BURST_IBI_all, BURST_BIN_all, sfreq_p2, pat_list = pickle.load(file)

# the sampling frequency should be identical for controls & patients
if sfreq_p!=sfreq_p2:
    raise RuntimeError('old and new patient data sampling frequencies not matching')
if sfreq_c!=sfreq_p2:
    raise RuntimeError('control & patient data sampling frequencies not matching')

def fwhm2sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))
fwhm_c=sfreq_c/smoothing
sigma_c = fwhm2sigma(fwhm_c)
fwhm_p=sfreq_p/smoothing
sigma_p = fwhm2sigma(fwhm_p)

BI_all.append(BURST_INFO_all)
BB_all.append(BURST_BIN_all)

BURST_info_off=[]
BURST_info_on=[]
BURST_bin_off=[]
BURST_bin_on=[]
for h in range(0, len(BI_all)):
    for i in range(0, len(BI_all[h])):
        BURST_info=BI_all[h][i]
        BURST_bin=BB_all[h][i]
        
        BURST_info_off.append(BURST_info[0])
        BURST_info_on.append(BURST_info[1])
        BURST_bin_off.append(BURST_bin[0])
        BURST_bin_on.append(BURST_bin[1])

def post_burst_windowing(data_burst_info, data_burst_bin, lim, lim2, sfreq, window):
    Mrat_bgb=list()
    Mrat_bglb=list()
    Mrat_bgsb=list()
    for h in range(0, len(data_burst_info)):
        burst_info=data_burst_info[h]
        bin_burst=data_burst_bin[h]
                    
        # set limits for burst duration limits (lim defined at beginning of script)
        cutoff=np.ceil(lim/1000*sfreq) ### --> multiply with sfreq to get value in data points
        cutoff3=np.ceil(lim2/1000*sfreq) ### --> multiply with sfreq to get value in data points
        
        # define trigger events based on beta burst information:
        # all bursts, long bursts and short bursts 
        stop_long=[]
        stop_short=[]
        stop_all=[]
    
        for k in range(1, len(burst_info[0])-1):
            if burst_info[2][k]>0:                                    # high amp signal
                boffset=burst_info[1][k]+burst_info[0][k]
                bdur=burst_info[0][k]
                if bdur >= cutoff:                                          # it is a burst, dur > lim 
                    stop_all=np.concatenate((stop_all,boffset), axis=None)           # all bursts
                    if burst_info[0][k]<=cutoff3:                 # short bursts
                        stop_short=np.concatenate((stop_short,boffset), axis=None)
                    else:                                               # long bursts
                        stop_long=np.concatenate((stop_long,boffset), axis=None)
            
        # binarized time series
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
            
        # make mean based on whole timeseries   
        baseline=np.mean(burst_binary_all)
    
        # calculate probabilities
        # observation window length
        ws=int(sfreq*window)
    
        # prob_bgb
        prob_bgb = list()
        for k in range(0, len(stop_all)):
            start=int(stop_all[k])
            stop=int(stop_all[k]+ws)
            tmp=burst_binary_all[start:stop]
            if len(tmp)==ws:
                prob_bgb.append(tmp)  
                
        # prob_bglb & prob_lbglb
        prob_bglb = list()
        for k in range(0, len(stop_long)):
            start=int(stop_long[k])
            stop=int(stop_long[k]+ws)
            tmp=burst_binary_all[start:stop]
            if len(tmp)==ws:
                prob_bglb.append(tmp)  
                
        # prob_bgsb 
        prob_bgsb = list()
        for k in range(0, len(stop_short)):
            start=int(stop_short[k])
            stop=int(stop_short[k]+ws)
            tmp=burst_binary_all[start:stop]
            if len(tmp)==ws:
                prob_bgsb.append(tmp)  
                
        # calculate means
        mprob_bgb=np.mean(prob_bgb, axis=0)
        mprob_bglb=np.mean(prob_bglb, axis=0)
        mprob_bgsb=np.mean(prob_bgsb, axis=0)
          
        # smooth raw curves
        mprob_bgb=scipy.ndimage.filters.gaussian_filter1d(mprob_bgb, sigma_c, axis=-1, order=0, mode='reflect', cval=0.0, truncate=4.0)
        mprob_bglb=scipy.ndimage.filters.gaussian_filter1d(mprob_bglb, sigma_c, axis=-1, order=0, mode='reflect', cval=0.0, truncate=4.0)
        mprob_bgsb=scipy.ndimage.filters.gaussian_filter1d(mprob_bgsb, sigma_c, axis=-1, order=0, mode='reflect', cval=0.0, truncate=4.0)
                
        # ratio post burst bursting probability/baseline bursting probability
        Mrat_bgb.append(mprob_bgb/baseline)
        Mrat_bglb.append(mprob_bglb/baseline)
        Mrat_bgsb.append(mprob_bgsb/baseline)
    
        ts_length=len(mprob_bgb)

    return Mrat_bgb, Mrat_bglb, Mrat_bgsb, ts_length

def get_mean_sem(Mrat_bgb, Mrat_bglb, Mrat_bgsb):
    MRAT_bgb=np.mean(Mrat_bgb, axis=0)
    MRAT_bglb=np.mean(Mrat_bglb, axis=0)
    MRAT_bgsb=np.mean(Mrat_bgsb, axis=0)
    
    SEMRAT_bgb=sem(Mrat_bgb, axis=0)
    SEMRAT_bglb=sem(Mrat_bglb, axis=0)
    SEMRAT_bgsb=sem(Mrat_bgsb, axis=0)
    
    return MRAT_bgb, MRAT_bglb, MRAT_bgsb, SEMRAT_bgb, SEMRAT_bglb, SEMRAT_bgsb

Mrat_bgb_C, Mrat_bglb_C, Mrat_bgsb_C, ts_c = post_burst_windowing(BURST_INFO_all_c, BURST_BIN_all_c, lim, lim2, sfreq_c, window)
MRAT_bgb_C, MRAT_bglb_C, MRAT_bgsb_C, SEMRAT_bgb_C, SEMRAT_bglb_C, SEMRAT_bgsb_C= get_mean_sem(Mrat_bgb_C, Mrat_bglb_C, Mrat_bgsb_C)

Mrat_bgb_OFF, Mrat_bglb_OFF, Mrat_bgsb_OFF, ts_pd = post_burst_windowing(BURST_info_off, BURST_bin_off, lim, lim2, sfreq_p, window)
MRAT_bgb_OFF, MRAT_bglb_OFF, MRAT_bgsb_OFF, SEMRAT_bgb_OFF, SEMRAT_bglb_OFF, SEMRAT_bgsb_OFF= get_mean_sem(Mrat_bgb_OFF, Mrat_bglb_OFF, Mrat_bgsb_OFF)

Mrat_bgb_ON, Mrat_bglb_ON, Mrat_bgsb_ON, ts_pd = post_burst_windowing(BURST_info_on, BURST_bin_on, lim, lim2, sfreq_p, window)
MRAT_bgb_ON, MRAT_bglb_ON, MRAT_bgsb_ON, SEMRAT_bgb_ON, SEMRAT_bglb_ON, SEMRAT_bgsb_ON= get_mean_sem(Mrat_bgb_ON, Mrat_bglb_ON, Mrat_bgsb_ON)

# smooth mean curves
MRAT_bgb_ON=scipy.ndimage.filters.gaussian_filter1d(MRAT_bgb_ON, sigma_p, axis=-1, order=0, mode='reflect', cval=0.0, truncate=4.0)
MRAT_bglb_ON=scipy.ndimage.filters.gaussian_filter1d(MRAT_bglb_ON, sigma_p, axis=-1, order=0, mode='reflect', cval=0.0, truncate=4.0)
MRAT_bgsb_ON=scipy.ndimage.filters.gaussian_filter1d(MRAT_bgsb_ON, sigma_p, axis=-1, order=0, mode='reflect', cval=0.0, truncate=4.0)

MRAT_bgb_OFF=scipy.ndimage.filters.gaussian_filter1d(MRAT_bgb_OFF, sigma_p, axis=-1, order=0, mode='reflect', cval=0.0, truncate=4.0)
MRAT_bglb_OFF=scipy.ndimage.filters.gaussian_filter1d(MRAT_bglb_OFF, sigma_p, axis=-1, order=0, mode='reflect', cval=0.0, truncate=4.0)
MRAT_bgsb_OFF=scipy.ndimage.filters.gaussian_filter1d(MRAT_bgsb_OFF, sigma_p, axis=-1, order=0, mode='reflect', cval=0.0, truncate=4.0)

MRAT_bgb_C=scipy.ndimage.filters.gaussian_filter1d(MRAT_bgb_C, sigma_c, axis=-1, order=0, mode='reflect', cval=0.0, truncate=4.0)
MRAT_bglb_C=scipy.ndimage.filters.gaussian_filter1d(MRAT_bglb_C, sigma_c, axis=-1, order=0, mode='reflect', cval=0.0, truncate=4.0)
MRAT_bgsb_C=scipy.ndimage.filters.gaussian_filter1d(MRAT_bgsb_C, sigma_c, axis=-1, order=0, mode='reflect', cval=0.0, truncate=4.0)

# x axis & labels
ticks_scaler=5
time_c=np.arange(0, ts_c)
xt_c=np.arange(0, ts_c, sfreq_c*ticks_scaler)
xlabel_c=np.arange(0, len(xt_c)*5, ticks_scaler)

time_p=np.arange(0, ts_pd)
xt_p=np.arange(0, ts_pd, sfreq_p*ticks_scaler)
xlabel_p=np.arange(0, len(xt_p)*5, ticks_scaler)

# create constant scaler to plot probability = 1
pb_c=list()
for i in range(0,ts_c):
    tmp=1
    pb_c.append(tmp)
    
pb_pd=list()
for i in range(0,ts_pd):
    tmp=1
    pb_pd.append(tmp)

####################################################################
##################### stats on ratio info ##########################
####################################################################

# make non-overlapping sliding window, then do one-sided, one sample t-test comparing 
# mean in window to one --> output 1 if significant, 0 if not

ws=1*sfreq_c # defining the size of each frame length (to do testing on) --> 1 = 1 second
n_pts = 10

def post_burst_ts_sig_test(window, ws, data, height):
    sig_info=[]
    n_pts = 10
    for i in range(0, window): # go through all time windows to end of post-burst observation period 'window'
        start=int(i*ws) # define time window start
        stop=int((i+1)*ws) # define time window end
        TMP=[]
        for j in range(0, len(data)): # go through all subjects and get mean value per subject
            TMP.append(np.mean(data[j][start:stop]))
        t1, p1=scipy.stats.ttest_1samp(TMP, 1, alternative ='greater') # test whether in this window, the subjects' distribution differs from baseline
        if p1 < 0.05:
            sig_info.append(1)
        else:
            sig_info.append(0)
            
    # go through siginificance variable to do neighborhood dependent 'smoothing' --> single significant values are removed
    TMP=[]
    TMP.append(sig_info[0])
    for i in range(1, len(sig_info)-1):
        if np.logical_and(sig_info[i-1]==0, sig_info[i+1]==0): # if both left and right neighbours of significant values are 0, set to 0 (i.e. nonsignificant)
            tmp=0
        else: # if either or both neighbours are significant, keep value as is
            tmp=sig_info[i]
        TMP.append(tmp)
    # same procedure for last value in series: if last value is significant but non-contingent with other significant periods, remove
    if np.logical_and(sig_info[-1]==1, sig_info[-2]==0): 
        tmp=0
    else:
        tmp=sig_info[-1]
    TMP.append(tmp)
    sig_info=TMP 
            
    x_for_plot=list()
    y_for_plot=list()
    for i in range(0, len(sig_info)):
        if sig_info[i]==1:
            x_for_plot.append(np.arange(i,i+1,1/n_pts))
            for i in range(0,n_pts):
                y_for_plot.append(height)
    if np.sum(x_for_plot)>0:
        x_for_plot=np.concatenate(x_for_plot)
    
    return sig_info, x_for_plot, y_for_plot

# burst given burst
sig_bgb_off, x1off, y1off= post_burst_ts_sig_test(window, ws, Mrat_bgb_OFF, 1.2)
sig_bgb_on, x1on, y1on= post_burst_ts_sig_test(window, ws, Mrat_bgb_ON, 1.2-0.05)
sig_bgb_c, x1c, y1c= post_burst_ts_sig_test(window, ws, Mrat_bgb_C, 1.2)

# burst given long burst
sig_bglb_off, x2off, y2off= post_burst_ts_sig_test(window, ws, Mrat_bglb_OFF, 1.22)
sig_bglb_on, x2on, y2on= post_burst_ts_sig_test(window, ws, Mrat_bglb_ON, 1.22-0.05)
sig_bglb_c, x2c, y2c= post_burst_ts_sig_test(window, ws, Mrat_bglb_C, 1.22)

# burst given short burst
sig_bgsb_off, x4off, y4off= post_burst_ts_sig_test(window, ws, Mrat_bgsb_OFF, 1.22)
sig_bgsb_on, x4on, y4on= post_burst_ts_sig_test(window, ws, Mrat_bgsb_ON, 1.22-0.05)
sig_bgsb_c, x4c, y4c= post_burst_ts_sig_test(window, ws, Mrat_bgsb_C, 1.22)

# load dispersion coeffcient data
with open(path_to_procdat + 'dispersion_coefficient.pkl', 'rb') as file:
    mvr_ibi_OFF, mvr_ibi_ON, mvr_ibi_ctrl = pickle.load(file)

########################################################################################################
########### STATISTICS #################################################################################
########################################################################################################
########### ratio plots per burst type #################################################################
########################################################################################################

p_off=[]
p_on=[]
for i in range(0, len(Mrat_bglb_ON)):
    p_on.append(np.max(Mrat_bgb_ON[i][200:400]))
    p_off.append(np.max(Mrat_bgb_OFF[i][200:400]))

p_c=[]
for i in range(0, len(Mrat_bglb_C)):
    p_c.append(np.max(Mrat_bgb_C[i][200:400]))
    
print('RE-BURST PROBABILITY, all bursts')
print(' ')
print('duration of peak elevation')
print('DBS OFF: %s' % np.sum(sig_bgb_off))
print('DBS ON: %s' % np.sum(sig_bgb_on))
print('control: %s' % np.sum(sig_bgb_c))

stat, p1= scipy.stats.normaltest(p_c)
stat, p2= scipy.stats.normaltest(p_on)
stat, p3= scipy.stats.normaltest(p_off)

if np.logical_or(p2<0.05, p3<0.05):
    t, p= scipy.stats.wilcoxon(p_off, y=p_on)
    print('Median peak DBSOFF: %s' % np.median(p_off))
    print('Median peak DBSON: %s' % np.median(p_on))
    print('ON vs. OFF, Wilcoxon test')
    print('t=%s, p=%s' % (t,p))
    print(' ')
else:
    t, p= scipy.stats.ttest_rel(p_off, p_on, axis=0)
    print('ON vs. OFF, rel samples t-test')
    print('Mean peak DBSOFF: %s' % np.mean(p_off))
    print('Mean peak DBSON: %s' % np.mean(p_on))
    print('t=%s, p=%s' % (t,p))
    print(' ')   
    
if np.logical_or(p1<0.05, p2<0.05):
    t, p=  scipy.stats.mannwhitneyu(p_on, p_c)
    print('Median peak DBSON: %s' % np.median(p_on))
    print('Median peak control: %s' % np.median(p_c))
    print('ON vs. control, Mann Whitney test')
    print('u=%s, p=%s' % (t,p))
    print(' ')
else:
    t, p= scipy.stats.ttest_ind(p_on, p_c, axis=0)
    print('Mean peak DBSON: %s' % np.mean(p_on))
    print('Mean peak control: %s' % np.mean(p_c))
    print('ON vs. control, ind. samples t-test')
    print('t=%s, p=%s' % (t,p))
    print(' ')   


if np.logical_or(p1<0.05, p3<0.05):
    t, p=  scipy.stats.mannwhitneyu(p_off, p_c)
    print('Median peak DBSOFF: %s' % np.median(p_off))
    print('Median peak control: %s' % np.median(p_c))
    print('OFF vs. control, Mann Whitney test')
    print('u=%s, p=%s' % (t,p))
    print(' ')
else:
    t, p= scipy.stats.ttest_ind(p_off, p_c, axis=0)
    print('Mean peak DBSOFF: %s' % np.mean(p_off))
    print('Mean peak control: %s' % np.mean(p_c))
    print('OFF vs. control, ind. samples t-test')
    print('t=%s, p=%s' % (t,p))
    print(' ')  
    
# labeling of boxplot outliers
def make_labels(ax, boxplot, data):
    for h in range(0, len(data)):
        # get indices of the outliers
        vals=boxplot['fliers'][h].get_data()[1]

        # Grab the relevant Line2D instances from the boxplot dictionary
        med = boxplot['medians'][h]
        fly = boxplot['fliers'][h]
    
        # The x position of the median line
        xpos = med.get_xdata()
    
        # Lets make the text have a horizontal offset which is some 
        # fraction of the width of the box
        xoff = 0.10 * (xpos[1] - xpos[0])
    
        # Many fliers, so we loop over them and create a label for each one
        i=0     
        for flier in fly.get_ydata():
            idx=np.where(data[h]==vals[i])
            i=i+1
            ax.text(1+h + xoff, flier,
                    '%s' % idx, va='center')
    
##############################################################################
########### Plotting #########################################################
##############################################################################
# figure fonts
font = {'family': 'arial',
                'color':  'black',
                'weight': 'normal',
                'size': 28,
                }
font2 = {'family': 'arial',
                'color':  'black',
                'weight': 'normal',
                'size': 24,
                }
font3 = {'family': 'arial',
                'color':  'black',
                'weight': 'normal',
                'size': 18,
                }

# colour specifications
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

n_pat_on_med=len(BI_all[0])

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
plt.subplots_adjust(bottom=0.15, left=0.1, right=0.95, top=0.9, wspace = 0.3)

ax1=plt.subplot(131)
ymin=0.8
ymax=2.2
mark = dict(marker='o')
data_ibi = [mvr_ibi_OFF, mvr_ibi_ON, mvr_ibi_ctrl]
boxes = ax1.boxplot(data_ibi, flierprops=mark)
make_labels(ax1, boxes, data_ibi)
numBoxes = len(data_ibi)
colour=['#377eb8', '#ff7f00', '#4daf4a'] # color scheme which is color blind friendly
# for i in range(0, numBoxes):
#     y = data_ibi[i]
#     x = np.random.normal(1+i, 0.04, size=len(y))
#     plt.plot(x, y, color=colour[i], marker='.', linestyle = 'None', alpha=0.2, markersize=14)
# for i in range(0, len(mvr_ibi_OFF)):
#     x=[1, 2]
#     y=[mvr_ibi_OFF[i], mvr_ibi_ON[i]]
#     plt.plot(x,y,'k', alpha=0.15, linewidth=0.8)
for i in range(0, numBoxes):
    y = data_ibi[i][n_pat_on_med:]
    x = np.random.normal(1+i, 0.04, size=len(y))
    plt.plot(x, y, color=colour[i], marker='.', linestyle = 'None', alpha=0.2, markersize=12)
for i in range(0, 2):
    y = data_ibi[i][0:n_pat_on_med]
    x=np.full((n_pat_on_med), i+1)
    plt.plot(x, y, color=colour[i], marker='*', linestyle = 'None', alpha=1, markersize=8)
for i in range(n_pat_on_med, len(mvr_ibi_OFF)):
    x=[1, 2]
    y=[mvr_ibi_OFF[i], mvr_ibi_ON[i]]
    plt.plot(x,y, 'k', alpha=0.2, linewidth=0.6)
for i in range(0, n_pat_on_med):
    x=[1, 2]
    y=[mvr_ibi_OFF[i], mvr_ibi_ON[i]]
    plt.plot(x,y, 'k', alpha=0.6, linewidth=0.6)    
plt.xticks([1, 2, 3], ['OFF', 'ON', 'ctrl'], fontsize=24)
plt.yticks(fontsize=24)
plt.text(1, 2.0, '____________', fontdict=font3)
plt.text(1.5, 2.0, '*', fontdict=font3)
plt.text(1, 2.1, '________________________', fontdict=font3)
plt.text(2, 2.1, '*', fontdict=font3)
plt.ylim(ymin, ymax)
plt.ylabel('burst dispersion',fontdict=font)
plt.title('A', fontdict=font, loc='left')

ax2=plt.subplot(132)
#plt.subplots_adjust(bottom=0.1, left=0.1, right=0.95, top=0.9)
ymin=0.9
ymax=1.40
mark = dict(marker='o')
data_peak = [p_off, p_on, p_c]
boxes = ax2.boxplot(data_peak, flierprops=mark)
make_labels(ax2, boxes, data_peak)
numBoxes = len(data_peak)
# for i in range(0, numBoxes):
#     y = data_peak[i]
#     x = np.random.normal(1+i, 0.04, size=len(y))
#     plt.plot(x, y, color=colour[i], marker='.', linestyle = 'None', alpha=0.2, markersize=14)
# for i in range(0, len(p_off)):
#     x=[1, 2]
#     y=[p_off[i], p_on[i]]
#     plt.plot(x,y,'k', alpha=0.15, linewidth=0.8)
for i in range(0, numBoxes):
    y = data_peak[i][n_pat_on_med:]
    x = np.random.normal(1+i, 0.04, size=len(y))
    plt.plot(x, y, color=colour[i], marker='.', linestyle = 'None', alpha=0.2, markersize=12)
for i in range(0, 2):
    y = data_peak[i][0:n_pat_on_med]
    x=np.full((n_pat_on_med), i+1)
    plt.plot(x, y, color=colour[i], marker='*', linestyle = 'None', alpha=1, markersize=8)
for i in range(n_pat_on_med, len(p_off)):
    x=[1, 2]
    y=[p_off[i], p_on[i]]
    plt.plot(x,y, 'k', alpha=0.2, linewidth=0.6)
for i in range(0, n_pat_on_med):
    x=[1, 2]
    y=[p_off[i], p_on[i]]
    plt.plot(x,y, 'k', alpha=0.6, linewidth=0.6)     
plt.xticks([1, 2, 3], ['OFF', 'ON', 'ctrl'], fontsize=24)
plt.yticks(fontsize=24)
plt.text(1, 1.37, '____________', fontdict=font3)
plt.text(1.5, 1.37, '*', fontdict=font3)
# plt.text(1, 1.32, '________________________', fontdict=font)
# plt.text(2, 1.32, '*', fontdict=font)
plt.ylabel('maximum re-burst likelihood',fontdict=font)
plt.ylim(ymin,ymax)
plt.title('B', fontdict=font, loc='left')

plt.subplot(133)
ylim=1.25
ymin=0.9
plt.plot(time_p, MRAT_bgb_OFF, color=colour[0], label='DBS OFF')    
plt.plot(time_p, MRAT_bgb_ON, color=colour[1], label='DBS ON')    
plt.plot(time_c, MRAT_bgb_C, color=colour[2], label='control') 
plt.plot(time_p, pb_pd, 'k')
plt.scatter(x1off*int(sfreq_p),y1off, color=colour[0], alpha=0.15, edgecolors=None)
plt.scatter(x1on*int(sfreq_p),y1on, color=colour[1], alpha=0.15, edgecolors=None)
plt.scatter(x1c*int(sfreq_p),np.subtract(y1c,0.025), color=colour[2], alpha=0.15, edgecolors=None)
plt.plot(time_p, MRAT_bgb_OFF+SEMRAT_bgb_OFF, color=colour[0], alpha=0.3) 
plt.plot(time_p, MRAT_bgb_OFF-SEMRAT_bgb_OFF, color=colour[0], alpha=0.3) 
plt.fill_between(time_p, MRAT_bgb_OFF, MRAT_bgb_OFF+SEMRAT_bgb_OFF, facecolor=colour[0], alpha=0.3)
plt.fill_between(time_p, MRAT_bgb_OFF, MRAT_bgb_OFF-SEMRAT_bgb_OFF, facecolor=colour[0], alpha=0.3)
plt.plot(time_p, MRAT_bgb_ON+SEMRAT_bgb_ON, color=colour[1], alpha=0.3) 
plt.plot(time_p, MRAT_bgb_ON-SEMRAT_bgb_ON, color=colour[1], alpha=0.3) 
plt.fill_between(time_p, MRAT_bgb_ON, MRAT_bgb_ON+SEMRAT_bgb_ON, facecolor=colour[1], alpha=0.3)
plt.fill_between(time_p, MRAT_bgb_ON, MRAT_bgb_ON-SEMRAT_bgb_ON, facecolor=colour[1], alpha=0.3)
plt.plot(time_c, MRAT_bgb_C+SEMRAT_bgb_C, color=colour[2], alpha=0.3) 
plt.plot(time_c, MRAT_bgb_C-SEMRAT_bgb_C, color=colour[2], alpha=0.3) 
plt.fill_between(time_c, MRAT_bgb_C, MRAT_bgb_C+SEMRAT_bgb_C, facecolor=colour[2], alpha=0.3)
plt.fill_between(time_c, MRAT_bgb_C, MRAT_bgb_C-SEMRAT_bgb_C, facecolor=colour[2], alpha=0.3)
plt.xticks(xt_p,xlabel_p)
plt.xlabel('time post burst (s)',fontdict=font)  
plt.ylabel('burst probability',fontdict=font)  
plt.yticks(fontsize=24)
plt.xticks(fontsize=24)
plt.ylim(ymin,ylim)
plt.title('C', fontdict=font, loc='left')
plt.legend(fontsize=16, bbox_to_anchor=(0.95, 0.26))

ftitle=(path_to_figs + 'Figure_5.png')          
fig.savefig(ftitle)  

########################################################################################################
########### STATISTICS #################################################################################
########################################################################################################
########### Compare peaks for post-burst likelihood, short vs. long bursts #############################
########################################################################################################

###############################
###  BURST DISPERSION ####
###############################
print(' ')
print('BURST DISPERSION')
print(' ')

# load dispersion coeffcient data
with open(path_to_procdat + 'dispersion_coefficient.pkl', 'rb') as file:
    mvr_ibi_OFF, mvr_ibi_ON, mvr_ibi_ctrl = pickle.load(file)

stat, p1= scipy.stats.normaltest(mvr_ibi_OFF)
stat, p2= scipy.stats.normaltest(mvr_ibi_ON)
stat, p3= scipy.stats.normaltest(mvr_ibi_ctrl)
if np.logical_or(p1<0.05, p2<0.05):
    stat, p=scipy.stats.wilcoxon(mvr_ibi_OFF, y=mvr_ibi_ON)
    print('OFF vs. ON wilcoxon test')
    print('median OFF %s' % np.median(mvr_ibi_OFF))
    print('median ON %s' % np.median(mvr_ibi_ON))
    print('stat=%s, p=%s' % (stat,p))
    print(' ')
else:
    t, p= scipy.stats.ttest_rel(mvr_ibi_OFF, mvr_ibi_ON, axis=0)
    print('OFF vs. ON rel samples t-test')
    print('mean OFF %s' % np.mean(mvr_ibi_OFF))
    print('mean ON %s' % np.mean(mvr_ibi_ON))
    print('t=%s, p=%s' % (t,p))
    print(' ')

if np.logical_or(p1<0.05, p3<0.05):
    t, p=  scipy.stats.mannwhitneyu(mvr_ibi_OFF, mvr_ibi_ctrl)
    print('OFF vs. ctrl mann whitney u')
    print('median OFF %s' % np.median(mvr_ibi_OFF))
    print('median ctrl %s' % np.median(mvr_ibi_ctrl))
    print('u=%s, p=%s' % (t,p))
    print(' ')
else:
    t, p= scipy.stats.ttest_ind(mvr_ibi_OFF, mvr_ibi_ctrl, axis=0)
    print('OFF vs. ctrl ind samples t-test')
    print('mean OFF %s' % np.mean(mvr_ibi_OFF))
    print('mean ctrl %s' % np.mean(mvr_ibi_ctrl))
    print('t=%s, p=%s' % (t,p))
    print(' ')

if np.logical_or(p2<0.05, p3<0.05):
    t, p= scipy.stats.mannwhitneyu(mvr_ibi_ON, mvr_ibi_ctrl)
    print('ON vs. ctrl mann whitney u')
    print('median ON %s' % np.median(mvr_ibi_ON))
    print('median ctrl %s' % np.median(mvr_ibi_ctrl))
    print('u=%s, p=%s' % (t,p))
    print(' ')
else:
    t, p= scipy.stats.ttest_ind(mvr_ibi_ON, mvr_ibi_ctrl, axis=0)
    print('ON vs. ctrl ind samples t-test')
    print('mean ON %s' % np.mean(mvr_ibi_ON))
    print('mean ctrl %s' % np.mean(mvr_ibi_ctrl))
    print('t=%s, p=%s' % (t,p))
    print(' ')



p_lb_off=[]
p_sb_off=[]
p_lb_on=[]
p_sb_on=[]
for i in range(0, len(Mrat_bglb_ON)):
    tmp1=np.max(Mrat_bglb_ON[i][200:400])
    tmp2=np.max(Mrat_bglb_OFF[i][200:400])
    tmp3=np.max(Mrat_bgsb_ON[i][200:400])
    tmp4=np.max(Mrat_bgsb_OFF[i][200:400])
    p_lb_on.append(tmp1)
    p_lb_off.append(tmp2)
    p_sb_on.append(tmp3)
    p_sb_off.append(tmp4)

p_lb_c=[]
p_sb_c=[]
for i in range(0, len(Mrat_bglb_C)):
    tmp1=np.max(Mrat_bglb_C[i][200:400])
    tmp2=np.max(Mrat_bgsb_C[i][200:400])
    p_lb_c.append(tmp1)
    p_sb_c.append(tmp2)
    
print('RE-BURST PROBABILITY, Long vs. short')
print(' ')

print('duration of peak elevation')
print('DBS OFF, long: %s' % np.sum(sig_bglb_off))
print('DBS ON, long: %s' % np.sum(sig_bglb_on))
print('control, long: %s' % np.sum(sig_bglb_c))
print('DBS OFF, short: %s' % np.sum(sig_bgsb_off))
print('DBS ON, short: %s' % np.sum(sig_bgsb_on))
print('control, short: %s' % np.sum(sig_bgsb_c))
print(' ')

stat, p1= scipy.stats.normaltest(p_lb_c)
stat, p2= scipy.stats.normaltest(p_sb_c)

if np.logical_or(p1<0.05, p2<0.05):
    t, p= scipy.stats.wilcoxon(p_lb_c, y=p_sb_c)
    print('Median control long: %s' % np.median(p_lb_c))
    print('Median control short: %s' % np.median(p_sb_c))
    print('Ctrl., long vs. short, Wilcoxon test')
    print('t=%s, p=%s' % (t,p))
    print(' ')
else:
    t, p= scipy.stats.ttest_rel(p_lb_c, p_sb_c, axis=0)
    print('control long: %s' % np.mean(p_lb_c))
    print('control short: %s' % np.mean(p_sb_c))
    print('Ctrl., long vs. short, rel samples t-test')
    print('t=%s, p=%s' % (t,p))
    print(' ')   
    
stat, p1= scipy.stats.normaltest(p_lb_off)
stat, p2= scipy.stats.normaltest(p_sb_off)

if np.logical_or(p1<0.05, p2<0.05):
    t, p= scipy.stats.wilcoxon(p_lb_off, y=p_sb_off)
    print('Median DBSOFF long: %s' % np.median(p_lb_off))
    print('Median DBSOFF short: %s' % np.median(p_sb_off))
    print('DBS OFF, long vs. short, Wilcoxon test')
    print('t=%s, p=%s' % (t,p))
    print(' ')
else:
    t, p= scipy.stats.ttest_rel(p_lb_off, p_sb_off, axis=0)
    print('DBSOFF long: %s' % np.mean(p_lb_off))
    print('DBSOFF short: %s' % np.mean(p_sb_off))
    print('DBS OFF, long vs. short, rel samples t-test')
    print('t=%s, p=%s' % (t,p))
    print(' ')   

stat, p1= scipy.stats.normaltest(p_lb_on)
stat, p2= scipy.stats.normaltest(p_sb_on)

if np.logical_or(p1<0.05, p2<0.05):
    t, p= scipy.stats.wilcoxon(p_lb_on, y=p_sb_on)
    print('Median DBSON long: %s' % np.median(p_lb_on))
    print('Median DBSON short: %s' % np.median(p_sb_on))
    print('DBS ON, long vs. short, Wilcoxon test')
    print('t=%s, p=%s' % (t,p))
    print(' ')
else:
    t, p= scipy.stats.ttest_rel(p_lb_on, p_sb_on, axis=0)
    print('DBSON long: %s' % np.mean(p_lb_on))
    print('DBSON short: %s' % np.mean(p_sb_on))
    print('DBS ON, long vs. short, rel samples t-test')
    print('t=%s, p=%s' % (t,p))
    print(' ')      

##############################################################################
########### Plotting #########################################################
##############################################################################

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
plt.subplots_adjust(bottom=0.15, left=0.1, right=0.95, top=0.9, wspace = 0.3)

ax1=plt.subplot(131)
ymin=0.8
ymax=2.2
mark = dict(marker='o')
data_ibi = [mvr_ibi_OFF, mvr_ibi_ON, mvr_ibi_ctrl]
boxes = ax1.boxplot(data_ibi, flierprops=mark)
make_labels(ax1, boxes, data_ibi)
numBoxes = len(data_ibi)
colour=['#377eb8', '#ff7f00', '#4daf4a'] # color scheme which is color blind friendly
for i in range(0, numBoxes):
    y = data_ibi[i][n_pat_on_med:]
    x = np.random.normal(1+i, 0.04, size=len(y))
    plt.plot(x, y, color=colour[i], marker='.', linestyle = 'None', alpha=0.2, markersize=12)
for i in range(0, 2):
    y = data_ibi[i][0:n_pat_on_med]
    x=np.full((n_pat_on_med), i+1)
    plt.plot(x, y, color=colour[i], marker='*', linestyle = 'None', alpha=1, markersize=8)
for i in range(n_pat_on_med, len(mvr_ibi_OFF)):
    x=[1, 2]
    y=[mvr_ibi_OFF[i], mvr_ibi_ON[i]]
    plt.plot(x,y, 'k', alpha=0.2, linewidth=0.6)
for i in range(0, n_pat_on_med):
    x=[1, 2]
    y=[mvr_ibi_OFF[i], mvr_ibi_ON[i]]
    plt.plot(x,y, 'k', alpha=0.6, linewidth=0.6)    
plt.xticks([1, 2, 3], ['OFF', 'ON', 'ctrl'], fontsize=24)
plt.yticks(fontsize=24)
plt.text(1, 2.0, '____________', fontdict=font3)
plt.text(1.5, 2.0, '*', fontdict=font3)
plt.text(1, 2.1, '________________________', fontdict=font3)
plt.text(2, 2.1, '*', fontdict=font3)
plt.ylim(ymin, ymax)
plt.ylabel('burst dispersion',fontdict=font)
plt.title('A', fontdict=font, loc='left')

ax2=plt.subplot(132)
ymin=0.9
ymax=1.40
mark = dict(marker='o')
data_peak = [p_off, p_on, p_c]
boxes = ax2.boxplot(data_peak, flierprops=mark)
make_labels(ax2, boxes, data_peak)
numBoxes = len(data_peak)
for i in range(0, numBoxes):
    y = data_peak[i][n_pat_on_med:]
    x = np.random.normal(1+i, 0.04, size=len(y))
    plt.plot(x, y, color=colour[i], marker='.', linestyle = 'None', alpha=0.2, markersize=12)
for i in range(0, 2):
    y = data_peak[i][0:n_pat_on_med]
    x=np.full((n_pat_on_med), i+1)
    plt.plot(x, y, color=colour[i], marker='*', linestyle = 'None', alpha=1, markersize=8)
for i in range(n_pat_on_med, len(p_off)):
    x=[1, 2]
    y=[p_off[i], p_on[i]]
    plt.plot(x,y, 'k', alpha=0.2, linewidth=0.6)
for i in range(0, n_pat_on_med):
    x=[1, 2]
    y=[p_off[i], p_on[i]]
    plt.plot(x,y, 'k', alpha=0.6, linewidth=0.6)     
plt.xticks([1, 2, 3], ['OFF', 'ON', 'ctrl'], fontsize=24)
plt.yticks(fontsize=24)
plt.text(1, 1.37, '____________', fontdict=font3)
plt.text(1.5, 1.37, '*', fontdict=font3)
# plt.text(1, 1.32, '________________________', fontdict=font)
# plt.text(2, 1.32, '*', fontdict=font)
plt.ylabel('maximum re-burst likelihood',fontdict=font)
plt.ylim(ymin,ymax)
plt.title('B', fontdict=font, loc='left')

plt.subplot(133)
ylim=1.25
ymin=0.9
plt.plot(time_p, MRAT_bgb_OFF, color=colour[0], label='DBS OFF')    
plt.plot(time_p, MRAT_bgb_ON, color=colour[1], label='DBS ON')    
plt.plot(time_c, MRAT_bgb_C, color=colour[2], label='control') 
plt.plot(time_p, pb_pd, 'k')
plt.scatter(x1off*int(sfreq_p),y1off, color=colour[0], alpha=0.15, edgecolors=None)
plt.scatter(x1on*int(sfreq_p),y1on, color=colour[1], alpha=0.15, edgecolors=None)
plt.scatter(x1c*int(sfreq_p),np.subtract(y1c,0.025), color=colour[2], alpha=0.15, edgecolors=None)
plt.plot(time_p, MRAT_bgb_OFF+SEMRAT_bgb_OFF, color=colour[0], alpha=0.3) 
plt.plot(time_p, MRAT_bgb_OFF-SEMRAT_bgb_OFF, color=colour[0], alpha=0.3) 
plt.fill_between(time_p, MRAT_bgb_OFF, MRAT_bgb_OFF+SEMRAT_bgb_OFF, facecolor=colour[0], alpha=0.3)
plt.fill_between(time_p, MRAT_bgb_OFF, MRAT_bgb_OFF-SEMRAT_bgb_OFF, facecolor=colour[0], alpha=0.3)
plt.plot(time_p, MRAT_bgb_ON+SEMRAT_bgb_ON, color=colour[1], alpha=0.3) 
plt.plot(time_p, MRAT_bgb_ON-SEMRAT_bgb_ON, color=colour[1], alpha=0.3) 
plt.fill_between(time_p, MRAT_bgb_ON, MRAT_bgb_ON+SEMRAT_bgb_ON, facecolor=colour[1], alpha=0.3)
plt.fill_between(time_p, MRAT_bgb_ON, MRAT_bgb_ON-SEMRAT_bgb_ON, facecolor=colour[1], alpha=0.3)
plt.plot(time_c, MRAT_bgb_C+SEMRAT_bgb_C, color=colour[2], alpha=0.3) 
plt.plot(time_c, MRAT_bgb_C-SEMRAT_bgb_C, color=colour[2], alpha=0.3) 
plt.fill_between(time_c, MRAT_bgb_C, MRAT_bgb_C+SEMRAT_bgb_C, facecolor=colour[2], alpha=0.3)
plt.fill_between(time_c, MRAT_bgb_C, MRAT_bgb_C-SEMRAT_bgb_C, facecolor=colour[2], alpha=0.3)
plt.xticks(xt_p,xlabel_p)
plt.xlabel('time post burst (s)',fontdict=font)  
plt.ylabel('burst probability',fontdict=font)  
plt.yticks(fontsize=24)
plt.xticks(fontsize=24)
plt.ylim(ymin,ylim)
plt.title('C', fontdict=font, loc='left')
plt.legend(fontsize=16, bbox_to_anchor=(0.95, 0.26))

ftitle=(path_to_figs + 'supplementary_figure_1.png')          
fig.savefig(ftitle)  

########################################################################################################
##################### STATISTICS #######################################################################
########################################################################################################

########################################################################################################
########### Compare peaks for post-burst likelihood, short vs. long bursts #############################
########################################################################################################

p_lb_off=[]
p_sb_off=[]
p_lb_on=[]
p_sb_on=[]
for i in range(0, len(Mrat_bglb_ON)):
    tmp1=np.max(Mrat_bglb_ON[i][200:400])
    tmp2=np.max(Mrat_bglb_OFF[i][200:400])
    tmp3=np.max(Mrat_bgsb_ON[i][200:400])
    tmp4=np.max(Mrat_bgsb_OFF[i][200:400])
    p_lb_on.append(tmp1)
    p_lb_off.append(tmp2)
    p_sb_on.append(tmp3)
    p_sb_off.append(tmp4)

p_lb_c=[]
p_sb_c=[]
for i in range(0, len(Mrat_bglb_C)):
    tmp1=np.max(Mrat_bglb_C[i][200:400])
    tmp2=np.max(Mrat_bgsb_C[i][200:400])
    p_lb_c.append(tmp1)
    p_sb_c.append(tmp2)
    
print('RE-BURST PROBABILITY, Long vs. short')
print(' ')

print('duration of peak elevation')
print('DBS OFF, long: %s' % np.sum(sig_bglb_off))
print('DBS ON, long: %s' % np.sum(sig_bglb_on))
print('control, long: %s' % np.sum(sig_bglb_c))
print('DBS OFF, short: %s' % np.sum(sig_bgsb_off))
print('DBS ON, short: %s' % np.sum(sig_bgsb_on))
print('control, short: %s' % np.sum(sig_bgsb_c))
print(' ')

stat, p1= scipy.stats.normaltest(p_lb_c)
stat, p2= scipy.stats.normaltest(p_sb_c)

if np.logical_or(p1<0.05, p2<0.05):
    t, p= scipy.stats.wilcoxon(p_lb_c, y=p_sb_c)
    print('Median control long: %s' % np.median(p_lb_c))
    print('Median control short: %s' % np.median(p_sb_c))
    print('Ctrl., long vs. short, Wilcoxon test')
    print('t=%s, p=%s' % (t,p))
    print(' ')
else:
    t, p= scipy.stats.ttest_rel(p_lb_c, p_sb_c, axis=0)
    print('control long: %s' % np.mean(p_lb_c))
    print('control short: %s' % np.mean(p_sb_c))
    print('Ctrl., long vs. short, rel samples t-test')
    print('t=%s, p=%s' % (t,p))
    print(' ')   

stat, p1= scipy.stats.normaltest(p_lb_off)
stat, p2= scipy.stats.normaltest(p_sb_off)

if np.logical_or(p1<0.05, p2<0.05):
    t, p= scipy.stats.wilcoxon(p_lb_off, y=p_sb_off)
    print('Median DBSOFF long: %s' % np.median(p_lb_off))
    print('Median DBSOFF short: %s' % np.median(p_sb_off))
    print('DBS OFF, long vs. short, Wilcoxon test')
    print('t=%s, p=%s' % (t,p))
    print(' ')
else:
    t, p= scipy.stats.ttest_rel(p_lb_off, p_sb_off, axis=0)
    print('DBSOFF long: %s' % np.mean(p_lb_off))
    print('DBSOFF short: %s' % np.mean(p_sb_off))
    print('DBS OFF, long vs. short, rel samples t-test')
    print('t=%s, p=%s' % (t,p))
    print(' ')   
    
stat, p1= scipy.stats.normaltest(p_lb_on)
stat, p2= scipy.stats.normaltest(p_sb_on)

if np.logical_or(p1<0.05, p2<0.05):
    t, p= scipy.stats.wilcoxon(p_lb_on, y=p_sb_on)
    print('Median DBSON long: %s' % np.median(p_lb_on))
    print('Median DBSON short: %s' % np.median(p_sb_on))
    print('DBS ON, long vs. short, Wilcoxon test')
    print('t=%s, p=%s' % (t,p))
    print(' ')
else:
    t, p= scipy.stats.ttest_rel(p_lb_on, p_sb_on, axis=0)
    print('DBSON long: %s' % np.mean(p_lb_on))
    print('DBSON short: %s' % np.mean(p_sb_on))
    print('DBS ON, long vs. short, rel samples t-test')
    print('t=%s, p=%s' % (t,p))
    print(' ')      

########################################################################################################
####################### PLOTTING #######################################################################
########################################################################################################

fig, axs = plt.subplots(1, 4, figsize=(24, 6))
plt.subplots_adjust(bottom=0.15, left=0.1, right=0.95, top=0.92, wspace=0.25)
colour=['#377eb8', '#ff7f00', '#4daf4a'] # color scheme which is color blind friendly

ylim=1.25
ymin=0.9

plt.subplot(141)
plt.plot(time_p, MRAT_bglb_OFF, colour[1])
plt.plot(time_p, MRAT_bgsb_OFF, colour[2])
plt.plot(time_p, pb_pd, 'k')
plt.scatter(x2off*int(sfreq_p),np.subtract(y2off, 0.02), color=colour[1], alpha=0.15, edgecolors=None)
plt.scatter(x4off*int(sfreq_p),np.subtract(y4off, 0.05), color=colour[2], alpha=0.15, edgecolors=None)
plt.plot(time_p, MRAT_bglb_OFF+SEMRAT_bglb_OFF, color=colour[1], alpha=0.3) 
plt.plot(time_p, MRAT_bglb_OFF-SEMRAT_bglb_OFF, color=colour[1], alpha=0.3) 
plt.fill_between(time_p, MRAT_bglb_OFF, MRAT_bglb_OFF+SEMRAT_bglb_OFF, facecolor=colour[1], alpha=0.3)
plt.fill_between(time_p, MRAT_bglb_OFF, MRAT_bglb_OFF-SEMRAT_bglb_OFF, facecolor=colour[1], alpha=0.3)
plt.plot(time_p, MRAT_bgsb_OFF+SEMRAT_bgsb_OFF, color=colour[2], alpha=0.3) 
plt.plot(time_p, MRAT_bgsb_OFF-SEMRAT_bgsb_OFF, color=colour[2], alpha=0.3) 
plt.fill_between(time_p, MRAT_bgsb_OFF, MRAT_bgsb_OFF+SEMRAT_bgsb_OFF, facecolor=colour[2], alpha=0.3)
plt.fill_between(time_p, MRAT_bgsb_OFF, MRAT_bgsb_OFF-SEMRAT_bgsb_OFF, facecolor=colour[2], alpha=0.3)
#plt.text(300, 1.18, '*', fontdict=font, fontsize=30)
plt.xticks(xt_p,xlabel_p)
plt.xlabel('time post burst (s)',fontdict=font)  
plt.ylabel('burst probability',fontdict=font)  
plt.yticks(fontsize=24)
plt.xticks(fontsize=24)
plt.ylim(ymin,ylim)
plt.title('A. DBS OFF',fontdict=font, loc='left')  

plt.subplot(142)
plt.plot(time_p, MRAT_bglb_ON, colour[1])
plt.plot(time_p, MRAT_bgsb_ON, colour[2])
plt.plot(time_p, pb_pd, 'k')
plt.scatter(x2on*int(sfreq_p),np.subtract(y2on, 0.02), color=colour[1], alpha=0.15, edgecolors=None)
plt.scatter(x4on*int(sfreq_p),np.subtract(y4on, 0.03), color=colour[2], alpha=0.15, edgecolors=None)
plt.plot(time_p, MRAT_bglb_ON+SEMRAT_bglb_ON, color=colour[1], alpha=0.3) 
plt.plot(time_p, MRAT_bglb_ON-SEMRAT_bglb_ON, color=colour[1], alpha=0.3) 
plt.fill_between(time_p, MRAT_bglb_ON, MRAT_bglb_ON+SEMRAT_bglb_ON, facecolor=colour[1], alpha=0.3)
plt.fill_between(time_p, MRAT_bglb_ON, MRAT_bglb_ON-SEMRAT_bglb_ON, facecolor=colour[1], alpha=0.3)
plt.plot(time_p, MRAT_bgsb_ON+SEMRAT_bgsb_ON, color=colour[2], alpha=0.3) 
plt.plot(time_p, MRAT_bgsb_ON-SEMRAT_bgsb_ON, color=colour[2], alpha=0.3) 
plt.fill_between(time_p, MRAT_bgsb_ON, MRAT_bgsb_ON+SEMRAT_bgsb_ON, facecolor=colour[2], alpha=0.3)
plt.fill_between(time_p, MRAT_bgsb_ON, MRAT_bgsb_ON-SEMRAT_bgsb_ON, facecolor=colour[2], alpha=0.3)
#plt.text(300, 1.15, '*', fontdict=font, fontsize=30)
plt.xticks(xt_p,xlabel_p)
plt.xlabel('time post burst (s)',fontdict=font)  
plt.yticks(fontsize=24)
plt.xticks(fontsize=24)
plt.ylim(ymin,ylim)
plt.title('B. DBS ON',fontdict=font, loc='left')  

plt.subplot(143)
data_peak = [p_lb_c, p_sb_c]

plt.plot(time_c, MRAT_bglb_C, colour[1], label='burst given long burst')
plt.plot(time_c, MRAT_bgsb_C, colour[2], label='burst given short burst')
plt.plot(time_c, pb_c, 'k-', label='baseline')
plt.scatter(x2c*int(sfreq_p),np.subtract(y2c, 0.02), color=colour[1], alpha=0.15, edgecolors=None)
plt.scatter(x4c*int(sfreq_p),np.subtract(y4c, 0.05), color=colour[2], alpha=0.15, edgecolors=None)
plt.plot(time_c, MRAT_bglb_C+SEMRAT_bglb_C, color=colour[1], alpha=0.3) 
plt.plot(time_c, MRAT_bglb_C-SEMRAT_bglb_C, color=colour[1], alpha=0.3) 
plt.fill_between(time_c, MRAT_bglb_C, MRAT_bglb_C+SEMRAT_bglb_C, facecolor=colour[1], alpha=0.3)
plt.fill_between(time_c, MRAT_bglb_C, MRAT_bglb_C-SEMRAT_bglb_C, facecolor=colour[1], alpha=0.3)
plt.plot(time_c, MRAT_bgsb_C+SEMRAT_bgsb_C, color=colour[2], alpha=0.3) 
plt.plot(time_c, MRAT_bgsb_C-SEMRAT_bgsb_C, color=colour[2], alpha=0.3) 
plt.fill_between(time_c, MRAT_bgsb_C, MRAT_bgsb_C+SEMRAT_bgsb_C, facecolor=colour[2], alpha=0.3)
plt.fill_between(time_c, MRAT_bgsb_C, MRAT_bgsb_C-SEMRAT_bgsb_C, facecolor=colour[2], alpha=0.3)
#plt.text(300, 1.18, '*', fontdict=font, fontsize=30)
plt.xticks(xt_p,xlabel_p)
plt.xlabel('time post burst (s)',fontdict=font)  
plt.yticks(fontsize=24)
plt.xticks(fontsize=24)
plt.ylim(ymin,ylim)
plt.title('C. control',fontdict=font, loc='left')  
plt.legend(fontsize=16)

ax2=plt.subplot(144)
ymin=0.85
ymax=1.70
data_peak = [p_lb_off, p_sb_off, p_lb_on, p_sb_on, p_lb_c, p_sb_c]
boxes = ax2.boxplot(data_peak, flierprops=mark)
make_labels(ax2, boxes, data_peak)
numBoxes = len(data_peak)

colour=['#377eb8', '#377eb8', '#ff7f00', '#ff7f00', '#4daf4a', '#4daf4a'] # color scheme which is color blind friendly
for i in range(0, numBoxes):
    y = data_peak[i]
    x = np.random.normal(1+i, 0.06, size=len(y))
    plt.plot(x, y, colour[i], marker='.', linestyle = 'None', alpha=0.2, markersize=12)
for i in range(0, 4):
    y = data_peak[i][0:n_pat_on_med]
    x = np.random.normal(1+i, 0.06, size=len(y))
    plt.plot(x, y, colour[i], marker='*', linestyle = 'None', alpha=1, markersize=8)
for i in range(n_pat_on_med, len(p_lb_off)): # patients OFF medication
    x=[1, 2]
    y=[p_lb_off[i], p_sb_off[i]]
    plt.plot(x,y,'k', alpha=0.2, linewidth=0.6)
    x1=[3, 4]
    y1=[p_lb_on[i], p_sb_on[i]]
    plt.plot(x1,y1,'k', alpha=0.2, linewidth=0.6)
for i in range(0, n_pat_on_med): # patients ON medication
    x=[1, 2]
    y=[p_lb_off[i], p_sb_off[i]]
    plt.plot(x,y,'k', alpha=0.6, linewidth=0.6)   
    x1=[3, 4]
    y1=[p_lb_on[i], p_sb_on[i]]
    plt.plot(x1,y1,'k', alpha=0.6, linewidth=0.6)   
for i in range(0, len(p_lb_c)): # controls
    x2=[5, 6]
    y2=[p_lb_c[i], p_sb_c[i]]
    plt.plot(x2,y2,'k', alpha=0.15, linewidth=0.8)  
plt.xticks([1, 2, 3, 4, 5, 6], ['long', 'short', 'long', 'short', 'long', 'short'], fontsize=24, rotation=45)
plt.yticks(fontsize=24)
plt.text(1, 1.45, '______', fontdict=font3)
plt.text(1.5, 1.45, '*', fontdict=font3)
plt.text(1.1, 1.5, 'OFF', fontdict=font2)

plt.text(3, 1.5, '______', fontdict=font3)
plt.text(3.5, 1.5, '*', fontdict=font3)
plt.text(3.27, 1.55, 'ON', fontdict=font2)

plt.text(5, 1.6, '______', fontdict=font3)
plt.text(5.5, 1.6, '*', fontdict=font3)
plt.text(4.9, 1.65, 'control', fontdict=font2)

plt.ylabel('maximum re-burst likelihood',fontdict=font)
plt.ylim(ymin,ymax)
plt.title('D. peak re-burst likelihood', fontdict=font, loc='left')  

ftitle=(path_to_figs + 'supplementary_figure_2.png')          
fig.savefig(ftitle)  
