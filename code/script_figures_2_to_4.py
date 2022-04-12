#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The file uses the output from basic burst analysis script to analyse basic
burst characteristics and compare them across conditions and groups. 

The script produces Figures 2, 3 and 4, and carries out the statistics that 
go with these figures.

It also makes a small pkl file, dispersioncoeff_output.pkl, which contains 
the dispersion coefficient data which is used in Figure 5. 


Parameters
----------
The file works on the output_file data for all three subject groups, 
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
Figure_2.png, Figure_3.png and Figure_4.png
statistics that go with these figures
file dispersioncoeff_output.pkl
--> contains dispersion coefficient data which is used in Figure 5. 

"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import sem
    
# general settings and information
lim=100
percentile=75
Cond=['OFF', 'ON']

# path information
path_trunc='../'
path_to_figs=path_trunc +'figures/'
path_to_rawdat=path_trunc+ 'raw_data/' 
path_to_procdat=path_trunc+ 'processed_data/' 

# Data file & path information
patients_old=path_to_procdat + 'old_patients.pkl' 
patients_new= path_to_procdat + 'new_patients.pkl'
controls=path_to_procdat + 'controls_all.pkl'
updrs_info=path_to_rawdat + 'updrs_hemibody_score.pkl'

# output file information
dispersioncoeff_output=path_to_procdat + 'dispersion_coefficient.pkl'

#if removing outliers:
#controls=path_to_procdat + 'controls_outliers_removed.pkl'
#dispersioncoeff_output='dispersion_coefficient_no_outliers.pkl'

##### LOAD PATIENT DATA
BR_all=[]
BD_all=[]
BA_all=[]
BBI_all=[]
IBI_all=[]

##### OLD DATA FIRST, n=3 with stable medication
with open(patients_old, 'rb') as file:  # Python 3: open(..., 'rb')
      BURST_RAW_all, BURST_DUR_all, BURST_AMP_all, BURST_INFO_all, BURST_BBI_all, BURST_IBI_all, BURST_BIN_all, sfreq_p, pat_list1 = pickle.load(file)

BR_all.append(BURST_RAW_all)
BD_all.append(BURST_DUR_all)
BA_all.append(BURST_AMP_all)
BBI_all.append(BURST_BBI_all)
IBI_all.append(BURST_IBI_all)

###### NEW DATA NEXT
with open(patients_new, 'rb') as file:  # Python 3: open(..., 'rb')
     BURST_RAW_all, BURST_DUR_all, BURST_AMP_all, BURST_INFO_all, BURST_BBI_all, BURST_IBI_all, BURST_BIN_all, sfreq_p2, pat_list2 = pickle.load(file)

BR_all.append(BURST_RAW_all)
BD_all.append(BURST_DUR_all)
BA_all.append(BURST_AMP_all)
BBI_all.append(BURST_BBI_all)
IBI_all.append(BURST_IBI_all)


# check sampling frequencies are identical (which should be the case for downsampled data)
if sfreq_p!=sfreq_p2:
    raise RuntimeError('old and new patient data sampling frequencies not matching')

# combine patient list information
pat_list1.append(pat_list2)
pat_list=np.hstack(pat_list1)

TMP=[]

all_MEAN_OFF_perc=[]
all_MEAN_ON_perc=[]

all_mbt_OFF=[]
all_mbt_ON=[]
all_bps_OFF_mean=[]
all_bps_ON_mean=[]
all_mba_OFF=[]
all_mba_ON=[]

mvr_ibi_OFF=[]
mvr_ibi_ON=[]

for h in range(0,len(BD_all)):
    for i in range(0, len(BD_all[h])):
        BURST_dur_ms=BD_all[h][i]
        BURST_amp=BA_all[h][i]
        BURST_raw=BR_all[h][i]
        BURST_ibi=IBI_all[h][i]
        BURST_bbi=BBI_all[h][i]

        ### Histogram of percentage of bursts in each bin, indiv. channels
        bins=[lim, 200, 300, 400, 500, 100000 ]
        OFF_dist, bin_edges=np.histogram(BURST_dur_ms[0], bins=bins)
        ON_dist, bin_edges=np.histogram(BURST_dur_ms[1], bins=bins)
        OFF_perc=OFF_dist/np.sum(OFF_dist)*100
        ON_perc=ON_dist/np.sum(ON_dist)*100
                    
        ### Histogram of percentage of bursts in each bin, all channels per subject
        all_MEAN_OFF_perc.append(OFF_perc)
        all_MEAN_ON_perc.append(ON_perc)
        
        # ### Dispersion coefficient
        bibi_ms_off=BURST_ibi[0]/sfreq_p*100
        mvr_ibi_OFF.append(np.sqrt(np.var(bibi_ms_off))/(np.mean(bibi_ms_off)))
        
        bibi_ms_on=BURST_ibi[1]/sfreq_p*100
        mvr_ibi_ON.append(np.sqrt(np.var(bibi_ms_on))/(np.mean(bibi_ms_on)))
        
        ### Mean bursting time ON vs OFF, all channels per subject        
        all_mbt_OFF.append(np.mean(BURST_dur_ms[0]))
        all_mbt_ON.append(np.mean(BURST_dur_ms[1]))
        
        ### Bursts/s, ON vs OFF, all channels per subject
        OFF_dist, bin_edges=np.histogram(BURST_dur_ms[0], bins=bins)
        bps_OFF=np.sum(OFF_dist)/(len(BURST_raw[0])/sfreq_p)
        
        ON_dist, bin_edges=np.histogram(BURST_dur_ms[1], bins=bins)
        bps_ON=np.sum(ON_dist)/(len(BURST_raw[1])/sfreq_p)
    
        all_bps_OFF_mean.append(bps_OFF)
        all_bps_ON_mean.append(bps_ON)
        
        ### Mean bursting amplitude ON vs OFF
        all_mba_OFF.append(np.mean(BURST_amp[0]))
        all_mba_ON.append(np.mean(BURST_amp[1]))

################################################################################################################
################################################################################################################
############################ CONTROL DATA ######################################################################
################################################################################################################
################################################################################################################

with open(controls, 'rb') as file:  # Python 3: open(..., 'rb')
     BURST_RAW_all_c, BURST_DUR_all_c, BURST_AMP_all_c, BURST_INFO_all_c, BURST_BBI_all_c, BURST_IBI_all_c, BURST_BIN_all_c, sfreq_c, control_list = pickle.load(file)
    
all_MEAN_ctrl_perc=[]
all_mbt_ctrl=[]
all_bps_ctrl_mean=[]
all_mba_ctrl=[]
mvr_ibi_ctrl=[]

for h in range(0, len(BURST_DUR_all_c)):
    BURST_dur_ms=BURST_DUR_all_c[h]
    BURST_amp=BURST_AMP_all_c[h]
    BURST_raw=BURST_RAW_all_c[h]
    BURST_ibi=BURST_IBI_all_c[h]
    BURST_bbi=BURST_BBI_all_c[h]
    
    ### Histogram of percentage of bursts in each bin, indiv. channels
    bins=[lim, 200, 300, 400, 500, 100000 ]

    ctrl_dist, bin_edges=np.histogram(BURST_dur_ms, bins=bins)
    ctrl_perc=ctrl_dist/np.sum(ctrl_dist)*100
    tb=np.sum(ctrl_dist)
    all_MEAN_ctrl_perc.append(ctrl_perc)    

    ### Dispersion coefficient
    bibi_ms_c=BURST_ibi/sfreq_c*100
    mvr_ibi_ctrl.append(np.sqrt(np.var(bibi_ms_c))/(np.mean(bibi_ms_c)))

    ### Mean bursting time ON vs OFF, all channels per subject
    all_mbt_ctrl.append(np.mean(BURST_dur_ms))
    
    ### Bursts/s, ON vs OFF, all channels per subject
    dist, bin_edges=np.histogram(BURST_dur_ms, bins=bins)
    bps_OFF=np.sum(dist)/(len(BURST_raw)/sfreq_c)
    all_bps_ctrl_mean.append(bps_OFF)
    
    ### Mean bursting amplitude ON vs OFF 
    all_mba_ctrl.append(np.mean(BURST_amp))
    
# get mean & sem values for plotting
def get_mean_and_sem(data):
    mean_data=np.mean(data, axis=0)
    sem_data=sem(data, axis=0)
    return mean_data, sem_data

# histogram data
MEAN_OFF_perc, SEM_OFF_perc = get_mean_and_sem(all_MEAN_OFF_perc)
MEAN_ON_perc, SEM_ON_perc = get_mean_and_sem(all_MEAN_ON_perc)
MEAN_ctrl_perc, SEM_ctrl_perc = get_mean_and_sem(all_MEAN_ctrl_perc)

# mean burst time
mbt_OFF, mbt_OFF_sem = get_mean_and_sem(all_mbt_OFF)
mbt_ON, mbt_ON_sem = get_mean_and_sem(all_mbt_ON)
mbt_ctrl, mbt_ctrl_sem = get_mean_and_sem(all_mbt_ctrl)

# mean burst amplitude
mba_OFF, mba_OFF_sem = get_mean_and_sem(all_mba_OFF)
mba_ON, mba_ON_sem = get_mean_and_sem(all_mba_ON)
mba_ctrl, mba_ctrl_sem = get_mean_and_sem(all_mba_ctrl)

# mean bursts/s
bps_OFF_mean, bps_OFF_sem = get_mean_and_sem(all_bps_OFF_mean)
bps_ON_mean, bps_ON_sem = get_mean_and_sem(all_bps_ON_mean)
bps_ctrl_mean, bps_ctrl_sem = get_mean_and_sem(all_bps_ctrl_mean)
        
# load UPDRS hemibody score information
with open(updrs_info, 'rb') as file:  # Python 3: open(..., 'rb')
      updrs_hemibody_scores = pickle.load(file)

# get only scores from table
HS=list()
for k in range(0, len(updrs_hemibody_scores)):
    tmp=updrs_hemibody_scores[k][1]
    HS.append(tmp)

change_mbt=100*(np.array(all_mbt_ON)-np.array(all_mbt_OFF))/np.array(all_mbt_OFF)
change_mba=100*(np.array(all_mba_ON)-np.array(all_mba_OFF))/np.array(all_mba_OFF)

ch_mbt=[]
ch_mba=[]
for j in range(0,len(change_mbt)):
    ch_mbt.append(change_mbt[j])
    ch_mba.append(change_mba[j])
        
################################################################################################################
################################################################################################################
################################ PLOTTING ######################################################################
################################################################################################################
################################################################################################################

### GENERAL PLOTTING SETTINGS
# number of hemispheres recorded ON medication
n_pat_on_med=len(BD_all[0])

# Font specifications
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

# routine enabling labelling of outliers in boxplots
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

#####################################################################
######## box plot duration/probability/amplitude ####################
#####################################################################

fig, axs = plt.subplots(1, 3, figsize=(19,8))
plt.subplots_adjust(bottom=0.1, left=0.08, right=0.98, top=0.9, wspace=0.28)
label=['OFF', 'ON', 'control']

ax1=plt.subplot(131)
ymin=160
ymax=320
mark = dict(marker='o')
data_dur = [all_mbt_OFF, all_mbt_ON, all_mbt_ctrl]
boxes = ax1.boxplot(data_dur, flierprops=mark)
make_labels(ax1, boxes, data_dur)
numBoxes = len(data_dur)
colour=['#377eb8', '#ff7f00', '#4daf4a'] # color scheme which is color blind friendly
for i in range(0, numBoxes):
    y = data_dur[i][n_pat_on_med:]
    x = np.random.normal(1+i, 0.04, size=len(y))
    plt.plot(x, y, color=colour[i], marker='.', linestyle = 'None', alpha=0.2, markersize=12)
for i in range(0, 2):
    y = data_dur[i][0:n_pat_on_med]
    x=np.full((n_pat_on_med), i+1)
    plt.plot(x, y, color=colour[i], marker='*', linestyle = 'None', alpha=1, markersize=8)
for i in range(n_pat_on_med, len(all_mbt_OFF)):
    x=[1, 2]
    y=[all_mbt_OFF[i], all_mbt_ON[i]]
    plt.plot(x,y, 'k', alpha=0.2, linewidth=0.6)
for i in range(0, len(all_mbt_OFF[0:n_pat_on_med])):
    x=[1, 2]
    y=[all_mbt_OFF[i], all_mbt_ON[i]]
    plt.plot(x,y, 'k', alpha=0.6, linewidth=0.6)
plt.xticks([1, 2, 3], ['OFF', 'ON', 'ctrl'], fontsize=24)
plt.yticks(fontsize=20)
plt.text(1, 290, '_________', fontdict=font3)
plt.text(1.5, 288, '*', fontdict=font3)
plt.text(1, 306, '__________________', fontdict=font3)
plt.text(2, 304, '*', fontdict=font3)
plt.ylabel('burst duration (ms)',fontdict=font)
plt.ylim(ymin, ymax)
plt.title('A. duration', fontdict=font)

ax3=plt.subplot(132) 
#ymin=0.9e-11
ymin=0
ymax=5.5e-11
data_amp = [all_mba_OFF, all_mba_ON, all_mba_ctrl]
boxes = ax3.boxplot(data_amp, flierprops=mark)
make_labels(ax3, boxes, data_amp)
for i in range(0, numBoxes):
    y = data_amp[i][n_pat_on_med:]
    x = np.random.normal(1+i, 0.04, size=len(y))
    plt.plot(x, y, color=colour[i], marker='.', linestyle = 'None', alpha=0.2, markersize=12)
for i in range(0, 2):
    y = data_amp[i][0:n_pat_on_med]
    x=np.full((n_pat_on_med), i+1)
    plt.plot(x, y, color=colour[i], marker='*', linestyle = 'None', alpha=1, markersize=8)
for i in range(n_pat_on_med, len(all_mba_OFF)):
    x=[1, 2]
    y=[all_mba_OFF[i], all_mba_ON[i]]
    plt.plot(x,y, 'k', alpha=0.2, linewidth=0.6)
for i in range(0, len(all_mba_OFF[0:n_pat_on_med])):
    x=[1, 2]
    y=[all_mba_OFF[i], all_mba_ON[i]]
    plt.plot(x,y, 'k', alpha=0.6, linewidth=0.6)
plt.xticks([1, 2, 3], ['OFF', 'ON', 'ctrl'],fontsize=24)
plt.yticks(fontsize=20)
plt.text(1, 4.7e-11, '_________', fontdict=font3)
plt.text(1.5, 4.65e-11, '*', fontdict=font3)
plt.text(1, 5.2e-11, '__________________', fontdict=font3)
plt.text(2, 5.15e-11, '*', fontdict=font3)
plt.text(2, 4.6e-11, '________', fontdict=font3)
plt.text(2.5, 4.55e-11, '*', fontdict=font3)
plt.ylabel('burst amplitude (T/cm)',fontdict=font)
plt.title('B. amplitude', fontdict=font)
plt.ylim(ymin, ymax)
plt.gcf().set_size_inches(15, 4)

ax2=plt.subplot(133) 
ymin=0.75
ymax=1.25
data_prob = [all_bps_OFF_mean, all_bps_ON_mean, all_bps_ctrl_mean]
boxes = ax2.boxplot(data_prob, flierprops=mark)
make_labels(ax2, boxes, data_prob)
for i in range(0, numBoxes):
    y = data_prob[i][n_pat_on_med:]
    x = np.random.normal(1+i, 0.04, size=len(y))
    plt.plot(x, y, color=colour[i], marker='.', linestyle = 'None', alpha=0.2, markersize=12)
for i in range(0, 2):
    y = data_prob[i][0:n_pat_on_med]
    x=np.full((n_pat_on_med), i+1)
    plt.plot(x, y, color=colour[i], marker='*', linestyle = 'None', alpha=1, markersize=8)
for i in range(n_pat_on_med, len(all_bps_OFF_mean)):
    x=[1, 2]
    y=[all_bps_OFF_mean[i], all_bps_ON_mean[i]]
    plt.plot(x,y, 'k', alpha=0.2, linewidth=0.6)
for i in range(0, len(all_bps_OFF_mean[0:n_pat_on_med])):
    x=[1, 2]
    y=[all_bps_OFF_mean[i], all_bps_ON_mean[i]]
    plt.plot(x,y, 'k', alpha=0.6, linewidth=0.6)
plt.xticks([1, 2, 3], ['OFF', 'ON', 'ctrl'],fontsize=24)
plt.yticks(fontsize=20)
plt.text(1, 1.2, '_________', fontdict=font3)
plt.text(1.5, 1.2, '*', fontdict=font3)
plt.ylabel('bursts/s',fontdict=font)
plt.ylim(ymin, ymax)
plt.title('C. burst rate', fontdict=font)

plt.show()
ftitle=(path_to_figs + 'Figure_2.png')          
fig.savefig(ftitle) 

#####################################################################
############## Histogram of burst durations, linscale ###############
#####################################################################

fig, ax1 = plt.subplots(1,1,
                          figsize=(8,8))
plt.subplots_adjust(bottom=0.1, left=0.15, right=0.95, top=0.95)
barWidth = 0.25 
ymax=80
r1 = np.arange(len(MEAN_OFF_perc)) # The x position of bars
r2 = [x + barWidth for x in r1]
r3 = [x + 2*barWidth for x in r1]
colour=['#377eb8', '#ff7f00', '#4daf4a'] # color scheme which is color blind friendly

bars1 = ax1.bar(r1, MEAN_OFF_perc, width = barWidth, color = colour[0], edgecolor = 'black', yerr=SEM_OFF_perc, capsize=7, label=Cond[0], alpha=0.4,)
bars3 = ax1.bar(r2, MEAN_ON_perc, width = barWidth, color = colour[1], edgecolor = 'black', yerr=SEM_ON_perc, capsize=7, label=Cond[1], alpha=0.4,)
bars5 = ax1.bar(r3, MEAN_ctrl_perc, width = barWidth, color = colour[2], edgecolor = 'black', yerr=SEM_ctrl_perc, capsize=7, label='control', alpha=0.4,)

numBoxes = len(MEAN_OFF_perc)
test=np.transpose(all_MEAN_OFF_perc)
for i in range(0, numBoxes):
    y = test[i]
    x = np.random.normal(r1[i], 0.03, size=len(y))
    plt.plot(x, y, color=colour[0], marker='.', linestyle = 'None', alpha=0.7, markersize=8)
    
test=np.transpose(all_MEAN_ON_perc)
for i in range(0, numBoxes):
    y = test[i]
    x = np.random.normal(r2[i], 0.03, size=len(y))
    plt.plot(x, y, color=colour[1], marker='.', linestyle = 'None', alpha=0.7, markersize=8)
    test=np.transpose(all_MEAN_OFF_perc)
    
test=np.transpose(all_MEAN_ctrl_perc)
for i in range(0, numBoxes):
    y = test[i]
    x = np.random.normal(r3[i], 0.03, size=len(y))
    plt.plot(x, y, color=colour[2], marker='.', linestyle = 'None', alpha=0.7, markersize=8)

ax1= plt.text(-0.1, 70, '____', fontdict=font3)
ax1= plt.text(0.125, 70, '*', fontdict=font3)
ax1= plt.text(-0.1, 75, '______', fontdict=font3)
ax1= plt.text(0.25, 75, '*', fontdict=font3)

ax1= plt.text(0.9, 30, '____', fontdict=font3)
ax1= plt.text(1.125, 30, '*', fontdict=font3)
ax1= plt.text(0.9, 35, '______', fontdict=font3)
ax1= plt.text(1.25, 35, '*', fontdict=font3)

ax1= plt.text(2.9, 7, '____', fontdict=font3)
ax1= plt.text(3.125, 7, '*', fontdict=font3)
ax1= plt.text(2.9, 12, '______', fontdict=font3)
ax1= plt.text(3.25, 12, '*', fontdict=font3)

ax1= plt.text(3.9, 7, '____', fontdict=font3)
ax1= plt.text(4.125, 7, '*', fontdict=font3)

xt=[r + barWidth for r in range(len(MEAN_OFF_perc))]
label=['0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '>0.5']
plt.xticks(xt,label, fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('% of total bursts',fontdict=font)
plt.xlabel('time window (s)',fontdict=font)
plt.legend(fontsize=20)
plt.ylim(0, ymax)

plt.show()
ftitle=(path_to_figs + 'Figure_3_linscale.png')          
fig.savefig(ftitle)  

#####################################################################
############## Histogram of burst durations, logscale ###############
#####################################################################

fig, ax1 = plt.subplots(1,1,
                          figsize=(8,8))
plt.subplots_adjust(bottom=0.1, left=0.15, right=0.95, top=0.95)
barWidth = 0.25 
ymax=120
r1 = np.arange(len(MEAN_OFF_perc)) # The x position of bars
r2 = [x + barWidth for x in r1]
r3 = [x + 2*barWidth for x in r1]
colour=['#377eb8', '#ff7f00', '#4daf4a'] # color scheme which is color blind friendly

bars1 = ax1.bar(r1, MEAN_OFF_perc, width = barWidth, color = colour[0], edgecolor = 'black', yerr=SEM_OFF_perc, capsize=7, label=Cond[0], alpha=0.4)
bars3 = ax1.bar(r2, MEAN_ON_perc, width = barWidth, color = colour[1], edgecolor = 'black', yerr=SEM_ON_perc, capsize=7, label=Cond[1], alpha=0.4)
bars5 = ax1.bar(r3, MEAN_ctrl_perc, width = barWidth, color = colour[2], edgecolor = 'black', yerr=SEM_ctrl_perc, capsize=7, label='control', alpha=0.4)

numBoxes = len(MEAN_OFF_perc)
test=np.transpose(all_MEAN_OFF_perc)
for i in range(0, numBoxes):
    y = test[i]
    x = np.random.normal(r1[i], 0.03, size=len(y))
    plt.plot(x, y, color=colour[0], marker='.', linestyle = 'None', alpha=0.7, markersize=8)
    
test=np.transpose(all_MEAN_ON_perc)
for i in range(0, numBoxes):
    y = test[i]
    x = np.random.normal(r2[i], 0.03, size=len(y))
    plt.plot(x, y, color=colour[1], marker='.', linestyle = 'None', alpha=0.7, markersize=8)
    test=np.transpose(all_MEAN_OFF_perc)
    
test=np.transpose(all_MEAN_ctrl_perc)
for i in range(0, numBoxes):
    y = test[i]
    x = np.random.normal(r3[i], 0.03, size=len(y))
    plt.plot(x, y, color=colour[2], marker='.', linestyle = 'None', alpha=0.7, markersize=8)
    
ax1= plt.text(-0.1, 80, '____', fontdict=font3)
ax1= plt.text(0.125, 80, '*', fontdict=font3)
ax1= plt.text(-0.1, 100, '______', fontdict=font3)
ax1= plt.text(0.25, 100, '*', fontdict=font3)

ax1= plt.text(0.9, 30, '____', fontdict=font3)
ax1= plt.text(1.125, 30, '*', fontdict=font3)
ax1= plt.text(0.9, 37, '______', fontdict=font3)
ax1= plt.text(1.25, 37, '*', fontdict=font3)

ax1= plt.text(2.9, 7, '____', fontdict=font3)
ax1= plt.text(3.125, 7, '*', fontdict=font3)
ax1= plt.text(2.9, 9, '______', fontdict=font3)
ax1= plt.text(3.25, 9, '*', fontdict=font3)

ax1= plt.text(3.9, 5, '____', fontdict=font3)
ax1= plt.text(4.125, 5, '*', fontdict=font3)

xt=[r + barWidth for r in range(len(MEAN_OFF_perc))]
label=['0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '>0.5']
plt.xticks(xt,label, fontsize=20)
plt.yticks(fontsize=20)
plt.yscale("log")
plt.ylabel('% of total bursts',fontdict=font)
plt.xlabel('time window (s)',fontdict=font)
plt.legend(fontsize=20)
plt.ylim(0, ymax)

plt.show()
ftitle=(path_to_figs + 'Figure_3_logscale.png')          
fig.savefig(ftitle)  

###########################################
### CORRELATION WITH HEMIBODY SCORE #######
###########################################

# UPDRS change & burst amplitude
import matplotlib.pyplot as plt

# Linear regression (fit first order polynomial)
import numpy.polynomial.polynomial as poly 
coefs = poly.polyfit(HS, ch_mba, 1)
ffit = poly.Polynomial(coefs)

x_new=np.linspace(20, 100, num=10)
fig=plt.figure(figsize=(10,8))
plt.subplots_adjust(bottom=0.15, left=0.15, right=0.95, top=0.95)
plt.scatter(HS[n_pat_on_med:],ch_mba[n_pat_on_med:], c=colour[0], s=50, alpha=0.4)
plt.scatter(HS[0:n_pat_on_med],ch_mba[0:n_pat_on_med], c=colour[1], s=50, alpha=0.4)
plt.plot(x_new, ffit(x_new), 'k', linewidth=2, alpha=0.5)
plt.yticks(fontsize=24)
plt.xticks(fontsize=24)
plt.xlabel('% reduction in cl. UPDRS hemibody score',fontdict=font)
plt.ylabel('% change in burst amplitude',fontdict=font)
plt.legend()

ftitle=(path_to_figs + 'Figure_4.png')          
fig.savefig(ftitle) 

################################################################################################################
################################################################################################################
################################ STATISTICS ####################################################################
################################################################################################################
################################################################################################################
 
########################################################
######### COMPARING ON AND OFF: ########################
######### TWO-WAY REPEATED MEASUSRES ANOVA #############
########################################################

###
# If you want to cite Pingouin, please use the publication in JOSS:
# Vallat, R. (2018). Pingouin: statistics in Python. Journal of Open Source Software, 3(31), 1026, https://doi.org/10.21105/joss.01026

import pandas as pd
import pingouin as pg
##########################
# data frame 1: Patients
##########################

# read data into one vector
test1=np.concatenate(all_MEAN_OFF_perc)
test2=np.concatenate(all_MEAN_ON_perc)


# QQ plots to assess normality
# percentages longer durations > 700 ms were not normally distributed
# these were pooled in the >500 ms category to avoid this count-related problem

label=['0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '>0.5']


TMP=[]
TMP.append(test1)
TMP.append(test2)
histodata=np.concatenate(TMP)

# get column data indices
n_pats=len(all_MEAN_OFF_perc)
n_bins=len(all_MEAN_OFF_perc[0])
n_cond=2

hemis=list()
for i in range(0,n_pats): 
    tmp=np.repeat(i, n_bins)
    hemis.append(tmp)
hemis=np.concatenate(hemis)

# make data indices
df1 = pd.DataFrame({'subj': np.tile(hemis,n_cond),
                    'DCond': np.repeat(['OFF', 'ON'], n_pats*n_bins),
                    'DBin': np.tile(['0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', 
                            '>0.5'], n_pats*n_cond),
                   'freq': histodata })

##########################
# data frame 2: Controls
##########################
n_pdpats=len(all_MEAN_OFF_perc)
n_pats=len(all_MEAN_ctrl_perc)
n_bins=len(all_MEAN_ctrl_perc[0])
n_cond=1
histodata=np.concatenate(all_MEAN_ctrl_perc)

hemis=list()
for i in range(n_pdpats,n_pats+n_pdpats): 
    tmp=np.repeat(i, n_bins)
    hemis.append(tmp)
hemis=np.concatenate(hemis)
    
# make data indices
df2 = pd.DataFrame({'subj': hemis,
                    'DCond': np.repeat(['control'], n_pats*n_bins),
                    'DBin': np.tile(['0.1-0.2', '0.2-0.3', '0.3-0.4','0.4-0.5', 
                            '>0.5'], n_pats*n_cond),
                    'freq': histodata })

df1_off = df1[df1['DCond'] == 'OFF']
df1_on = df1[df1['DCond'] == 'ON']
df_all = df1.append(df2)

#####################################################
# perform two-way repeated measures ANOVA, OFF vs. ON
#####################################################
from pingouin import rm_anova
aov = rm_anova(dv='freq', within=['DCond', 'DBin'],
                    subject='subj', data=df1, detailed=True, correction=True)
print('')
print('two way repeated measures ANOVA, OFF vs. ON')
pg.print_table(aov)


if aov.iloc[2][6] < 0.05:
    print('significant')
    posthocs = pg.pairwise_ttests(data=df1, dv='freq', between=None, within=['DBin', 'DCond'], subject='subj', alpha=0.05, padjust='fdr_bh', parametric=True)
    c=len(posthocs)
    ic=len(all_MEAN_OFF_perc[0])
    idx=c-ic
    pg.print_table(posthocs[idx:])
    
# interpretation of effect sizes (roughly):
# h = 0.20: small effect size
# h = 0.50: medium effect size
# h = 0.80: large effect size
    
####################################################
# perform two-way mixed model ANOVA, OFF vs. control
####################################################
df = df1_off.append(df2)
from pingouin import mixed_anova
aov = mixed_anova(dv='freq', between='DCond', within='DBin', subject='subj', data=df)
aov.round(3)
print('')
print('two way mixed model ANOVA, OFF vs. control')
pg.print_table(aov)


if aov.iloc[2][6] < 0.05:
    posthocs = pg.pairwise_ttests(data=df, dv='freq', between='DCond', within='DBin', subject='subj', alpha=0.05, padjust='fdr_bh', parametric=True)
    c=len(posthocs)
    ic=len(all_MEAN_OFF_perc[0])
    idx=c-ic
    pg.print_table(posthocs[idx:])
    
####################################################
# perform two-way mixed model ANOVA, ON vs. control
####################################################
df = df1_on.append(df2)
from pingouin import mixed_anova
aov = mixed_anova(dv='freq', between='DCond', within='DBin', subject='subj', data=df)
aov.round(3)
print('')
print('two way mixed model ANOVA, ON vs. control')
pg.print_table(aov)

if aov.iloc[2][6] < 0.05:
    posthocs = pg.pairwise_ttests(data=df, dv='freq', between='DCond', within='DBin', subject='subj', alpha=0.05, padjust='fdr_bh', parametric=False)
    c=len(posthocs)
    ic=len(all_MEAN_OFF_perc[0])
    idx=c-ic
    pg.print_table(posthocs[idx:])

########################################################
######### Comparing other burst statistics #############
########################################################
pvals=list()
comparison=list()

print(' ')
print('BURST DURATION')
print(' ')

stat, p1= scipy.stats.normaltest(all_mbt_OFF)
stat, p2= scipy.stats.normaltest(all_mbt_ON)
stat, p3= scipy.stats.normaltest(all_mbt_ctrl)
if np.logical_or(p1<0.05, p2<0.05):
    stat, p=scipy.stats.wilcoxon(all_mbt_OFF, y=all_mbt_ON)
    print('OFF vs. ON wilcoxon test')
    print('median OFF %s' % np.median(all_mbt_OFF))
    print('median ON %s' % np.median(all_mbt_ON))
    print('stat=%s, p=%s' % (stat,p))
    print(' ')
else:
    t, p= scipy.stats.ttest_rel(all_mbt_OFF, all_mbt_ON, axis=0)
    print('OFF vs. ON rel samples t-test')
    print('mean OFF %s' % np.mean(all_mbt_OFF))
    print('mean ON %s' % np.mean(all_mbt_ON))
    print('t=%s, p=%s' % (t,p))
    print(' ')
pvals.append(p)
comparison.append('bdur, OFF vs. ON')

if np.logical_or(p1<0.05, p3<0.05):
    t, p=  scipy.stats.mannwhitneyu(all_mbt_OFF, all_mbt_ctrl)
    print('OFF vs. ctrl mann whitney u')
    print('median OFF %s' % np.median(all_mbt_OFF))
    print('median ctrl %s' % np.median(all_mbt_ctrl))
    print('u=%s, p=%s' % (t,p))
    print(' ')
else:
    t, p= scipy.stats.ttest_ind(all_mbt_OFF, all_mbt_ctrl, axis=0)
    print('OFF vs. ctrl ind samples t-test')
    print('mean OFF %s' % np.mean(all_mbt_OFF))
    print('mean ctrl %s' % np.mean(all_mbt_ctrl))
    print('t=%s, p=%s' % (t,p))
    print(' ')
pvals.append(p)
comparison.append('bdur, OFF vs. ctrl')

if np.logical_or(p2<0.05, p3<0.05):
    t, p= scipy.stats.mannwhitneyu(all_mbt_ON, all_mbt_ctrl)
    print('ON vs. ctrl mann whitney u')
    print('median ON %s' % np.median(all_mbt_ON))
    print('median ctrl %s' % np.median(all_mbt_ctrl))
    print('u=%s, p=%s' % (t,p))
    print(' ')
else:
    t, p= scipy.stats.ttest_ind(all_mbt_ON, all_mbt_ctrl, axis=0)
    print('ON vs. ctrl ind samples t-test')
    print('mean ON %s' % np.mean(all_mbt_ON))
    print('mean ctrl %s' % np.mean(all_mbt_ctrl))
    print('t=%s, p=%s' % (t,p))
    print(' ')
pvals.append(p)
comparison.append('bdur, ON vs. ctrl')

#################################       
### BURST PROBABILITY #####
#################################
print(' ')
print('BURST PROBABILITY')
print(' ')

stat, p1= scipy.stats.normaltest(all_bps_OFF_mean)
stat, p2= scipy.stats.normaltest(all_bps_ON_mean)
stat, p3= scipy.stats.normaltest(all_bps_ctrl_mean)
if np.logical_or(p1<0.05, p2<0.05):
    stat, p=scipy.stats.wilcoxon(all_bps_OFF_mean, y=all_bps_ON_mean)
    print('OFF vs. ON wilcoxon test')
    print('median OFF %s' % np.median(all_bps_OFF_mean))
    print('median ON %s' % np.median(all_bps_ON_mean))
    print('stat=%s, p=%s' % (stat,p))
    print(' ')
else:
    t, p= scipy.stats.ttest_rel(all_bps_OFF_mean, all_bps_ON_mean, axis=0)
    print('OFF vs. ON rel samples t-test')
    print('mean OFF %s' % np.mean(all_bps_OFF_mean))
    print('mean ON %s' % np.mean(all_bps_ON_mean))
    print('t=%s, p=%s' % (t,p))
    print(' ')
pvals.append(p)
comparison.append('bps, OFF vs. ON')

if np.logical_or(p1<0.05, p3<0.05):
    t, p=  scipy.stats.mannwhitneyu(all_bps_OFF_mean, all_bps_ctrl_mean)
    print('OFF vs. ctrl mann whitney u')
    print('median OFF %s' % np.median(all_bps_OFF_mean))
    print('median ctrl %s' % np.median(all_bps_ctrl_mean))
    print('u=%s, p=%s' % (t,p))
    print(' ')
else:
    t, p= scipy.stats.ttest_ind(all_bps_OFF_mean, all_bps_ctrl_mean, axis=0)
    print('OFF vs. ctrl ind samples t-test')
    print('mean OFF %s' % np.mean(all_bps_OFF_mean))
    print('mean ctrl %s' % np.mean(all_bps_ctrl_mean))
    print('t=%s, p=%s' % (t,p))
    print(' ')
pvals.append(p)
comparison.append('bps, OFF vs. ctrl')

if np.logical_or(p2<0.05, p3<0.05):
    t, p= scipy.stats.mannwhitneyu(all_bps_ON_mean, all_bps_ctrl_mean)
    print('ON vs. ctrl mann whitney u')
    print('median ON %s' % np.median(all_bps_ON_mean))
    print('median ctrl %s' % np.median(all_bps_ctrl_mean))
    print('u=%s, p=%s' % (t,p))
    print(' ')
else:
    t, p= scipy.stats.ttest_ind(all_bps_ON_mean, all_bps_ctrl_mean, axis=0)
    print('ON vs. ctrl ind samples t-test')
    print('mean ON %s' % np.mean(all_bps_ON_mean))
    print('mean ctrl %s' % np.mean(all_bps_ctrl_mean))
    print('t=%s, p=%s' % (t,p))
    print(' ')
pvals.append(p)
comparison.append('bps, ON vs. ctrl')

###############################
### BURST amplitude #####
###############################
print(' ')
print('BURST AMPLITUDE')
print(' ')

stat, p1= scipy.stats.normaltest(all_mba_OFF)
stat, p2= scipy.stats.normaltest(all_mba_ON)
stat, p3= scipy.stats.normaltest(all_mba_ctrl)
if np.logical_or(p1<0.05, p2<0.05):
    stat, p=scipy.stats.wilcoxon(all_mba_OFF, y=all_mba_ON)
    print('OFF vs. ON wilcoxon test')
    print('median OFF %s' % np.median(all_mba_OFF))
    print('median ON %s' % np.median(all_mba_ON))
    print('stat=%s, p=%s' % (stat,p))
    print(' ')
else:
    t, p= scipy.stats.ttest_rel(all_mba_OFF, all_mba_ON, axis=0)
    print('OFF vs. ON rel samples t-test')
    print('mean OFF %s' % np.mean(all_mba_OFF))
    print('mean ON %s' % np.mean(all_mba_ON))
    print('t=%s, p=%s' % (t,p))
    print(' ')
pvals.append(p)
comparison.append('bamp, OFF vs. ON')

if np.logical_or(p1<0.05, p3<0.05):
    t, p=  scipy.stats.mannwhitneyu(all_mba_OFF, all_mba_ctrl)
    print('OFF vs. ctrl mann whitney u')
    print('median OFF %s' % np.median(all_mba_OFF))
    print('median ctrl %s' % np.median(all_mba_ctrl))
    print('u=%s, p=%s' % (t,p))
    print(' ')
else:
    t, p= scipy.stats.ttest_ind(all_mba_OFF, all_mba_ctrl, axis=0)
    print('OFF vs. ctrl ind samples t-test')
    print('mean OFF %s' % np.mean(all_mba_OFF))
    print('mean ctrl %s' % np.mean(all_mba_ctrl))
    print('t=%s, p=%s' % (t,p))
    print(' ')
pvals.append(p)
comparison.append('bamp, OFF vs. ctrl')

if np.logical_or(p2<0.05, p3<0.05):
    t, p= scipy.stats.mannwhitneyu(all_mba_ON, all_mba_ctrl)
    print('ON vs. ctrl mann whitney u')
    print('median ON %s' % np.median(all_mba_ON))
    print('median ctrl %s' % np.median(all_mba_ctrl))
    print('u=%s, p=%s' % (t,p))
    print(' ')
else:
    t, p= scipy.stats.ttest_ind(all_mba_ON, all_mba_ctrl, axis=0)
    print('ON vs. ctrl ind samples t-test')
    print('mean ON %s' % np.mean(all_mba_ON))
    print('mean ctrl %s' % np.mean(all_mba_ctrl))
    print('t=%s, p=%s' % (t,p))
    print(' ')
pvals.append(p)
comparison.append('bamp, ON vs. ctrl')

###########################################
### CORRELATION WITH HEMIBODY SCORE #######
###########################################

# UPDRS change & burst amplitude
import matplotlib.pyplot as plt

x_new=np.linspace(20, 100, num=10)

statistic1=scipy.stats.spearmanr(HS, b=ch_mba)
pvals.append(statistic1[1])
comparison.append('corr bamp-clinical')

print('Spearman correlation UPDRS hemibody & burst amplitude')
print(statistic1)
print(' ')

#################################################################
############# MULTIPLE COMPARISON CORRECTION ####################
############# Benjamini-Hochberg procedure ######################
#################################################################
from pingouin import multicomp

p_corr=multicomp(pvals, alpha=0.05, method='fdr_bh')
print('')
print('Benjamini-Hochberg corrected p values')
for i in range(0,len(comparison)):
    print('%s, %s, %s' % (comparison[i], p_corr[1][i], p_corr[0][i]))

# save dispersion coefficient info for use in burst probability script
# Saving the objects
with open(dispersioncoeff_output, 'wb') as f:
    pickle.dump([mvr_ibi_OFF, mvr_ibi_ON, mvr_ibi_ctrl], f)

