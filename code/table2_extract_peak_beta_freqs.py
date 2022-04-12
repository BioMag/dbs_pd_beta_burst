#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 14:49:35 2021

@author: amande
"""
import pickle
import sys
import numpy as np
from scipy.stats import sem
import scipy as scipy

# path information
path_trunc='../'
path_to_figs=path_trunc +'figures/'
path_to_procdat=path_trunc+ 'processed_data/'

# Data file & path information
patients_old=path_to_procdat + 'old_patients_psd_spectra.pkl' 
patients_new= path_to_procdat + 'new_patients_psd_spectra.pkl'
controls=path_to_procdat + 'controls_all_psd_spectra.pkl'

# load information about peak beta channels & peak beta frequencies
import info_beta_freqs_chans
beta_info_f_ch=info_beta_freqs_chans.info_beta_freqs_chans()
  
def get_beta_maximum(psd,f, beta_lo, beta_hi):
    beta_lo=int(beta_lo)
    beta_hi=int(beta_hi)
    idx=np.asarray(np.where(np.logical_and(f>int(beta_lo), f<int(beta_hi))))
    psd=np.squeeze(psd)
    beta_max_amplitude=np.max(psd[idx])
    idx_max_freq=np.where(psd==beta_max_amplitude)
    beta_max_freq=f[idx_max_freq]   
    return beta_max_freq

# control subjects
pat_list=['sib24', 'sib26', 'sib85', 'sib124', 'sib155', 'sib212','s09', 's10', 's11', 's15',  's16', 's17', 's18', 's19', 's20', 's21', 's22', 's23', 's24', 's25', 's26']
with open(controls, 'rb') as file:  # Python 3: open(..., 'rb')
     PSD, FREQUENCIES, pat_list_file = pickle.load(file)

beta_max=list()
beta_max_l=list()
beta_max_r=list()
for g in range(0, len(pat_list)):
    pat_id=pat_list[g]
    if pat_id!=pat_list_file[g]:
        sys.exit('patient id not matching, aborted')
    for j in range(0, len(beta_info_f_ch)):
        if beta_info_f_ch[j][0]==pat_id:
            for j in range(0, len(beta_info_f_ch)):
                if beta_info_f_ch[j][0]==pat_id:
                    psd=PSD[g]
                    f=FREQUENCIES[g][0]
                    
                    tmp=list()
                    if beta_info_f_ch[j][1]!='NaN':
                        beta_lo=beta_info_f_ch[j][1]
                        beta_hi=beta_info_f_ch[j][2]
                        beta_max_freq_l = get_beta_maximum(psd[0], f, beta_lo, beta_hi)
                        beta_max_l.append(beta_max_freq_l)
                        tmp.append(beta_max_freq_l)
                    if beta_info_f_ch[j][4]!='NaN':
                        beta_lo2=beta_info_f_ch[j][4]
                        beta_hi2=beta_info_f_ch[j][5]                        
                        beta_max_freq_r = get_beta_maximum(psd[1], f, beta_lo2, beta_hi2)
                        beta_max_r.append(beta_max_freq_r)
                        tmp.append(beta_max_freq_r)
                    beta_max.append(np.mean(tmp))
# new patients
pat_list=[5815, 5826, 5832, 5874, 5875, 5912, 5926, 5938, 5952, 5953, 5962, 5976, 5987]
with open(patients_new, 'rb') as file:  # Python 3: open(..., 'rb')
     PSD, FREQUENCIES, pat_list_file = pickle.load(file)

beta_max_patients=list()
beta_max_l_patients=list()
beta_max_r_patients=list()
for g in range(0, len(pat_list)):
    pat_id=pat_list[g]
    if pat_id!=pat_list_file[g]:
        sys.exit('patient id not matching, aborted')
    for j in range(0, len(beta_info_f_ch)):
        if beta_info_f_ch[j][0]==pat_id:
            for j in range(0, len(beta_info_f_ch)):
                if beta_info_f_ch[j][0]==pat_id:
                    psd=PSD[g][0]
                    f=FREQUENCIES[g][0]
                    tmp=list()
                    if beta_info_f_ch[j][1]!='NaN':
                        beta_lo=beta_info_f_ch[j][1]
                        beta_hi=beta_info_f_ch[j][2]
                        beta_max_freq_l = get_beta_maximum(psd, f, beta_lo, beta_hi)
                        beta_max_l_patients.append(beta_max_freq_l)
                        tmp.append(beta_max_freq_l)
                    if beta_info_f_ch[j][4]!='NaN':
                        beta_lo2=beta_info_f_ch[j][4]
                        beta_hi2=beta_info_f_ch[j][5]                        
                        beta_max_freq_r = get_beta_maximum(psd, f, beta_lo2, beta_hi2)
                        beta_max_r_patients.append(beta_max_freq_r)
                        tmp.append(beta_max_freq_r)
                    beta_max_patients.append(np.mean(tmp))                    
# old patients
pat_list=[4033, 4227, 4429]
with open(patients_old, 'rb') as file:  # Python 3: open(..., 'rb')
     PSD, FREQUENCIES, pat_list_file = pickle.load(file)

for g in range(0, len(pat_list)):
    pat_id=pat_list[g]
    if pat_id!=pat_list_file[g]:
        sys.exit('patient id not matching, aborted')
    for j in range(0, len(beta_info_f_ch)):
        if beta_info_f_ch[j][0]==pat_id:
            for j in range(0, len(beta_info_f_ch)):
                if beta_info_f_ch[j][0]==pat_id:
                    psd=PSD[g][0]
                    f=FREQUENCIES[g][0]
                    tmp=list()
                    if beta_info_f_ch[j][1]!='NaN':
                        beta_lo=beta_info_f_ch[j][1]
                        beta_hi=beta_info_f_ch[j][2]
                        beta_max_freq_l = get_beta_maximum(psd, f, beta_lo, beta_hi)
                        beta_max_l_patients.append(beta_max_freq_l)
                        tmp.append(beta_max_freq_l)
                    if beta_info_f_ch[j][4]!='NaN':
                        beta_lo2=beta_info_f_ch[j][4]
                        beta_hi2=beta_info_f_ch[j][5]                        
                        beta_max_freq_r = get_beta_maximum(psd, f, beta_lo2, beta_hi2)
                        beta_max_r_patients.append(beta_max_freq_r)
                        tmp.append(beta_max_freq_r)
                    beta_max_patients.append(np.mean(tmp))  
                    
# get means, sem and std
mean_beta_c=np.mean(beta_max)
sem_beta_c=sem(beta_max)
std_beta_c=np.std(beta_max)

mean_beta_c_l=np.mean(beta_max_l)
sem_beta_c_l=sem(beta_max_l)
std_beta_c_l=np.std(beta_max_l)

mean_beta_c_r=np.mean(beta_max_r)
sem_beta_c_r=sem(beta_max_r)
std_beta_c_r=np.std(beta_max_r)

mean_beta_p=np.mean(beta_max_patients)
sem_beta_p=sem(beta_max_patients)
std_beta_p=np.std(beta_max_patients)

mean_beta_p_l=np.mean(beta_max_l_patients)
sem_beta_p_l=sem(beta_max_l_patients)
std_beta_p_l=np.std(beta_max_l_patients)

mean_beta_p_r=np.mean(beta_max_r_patients)
sem_beta_p_r=sem(beta_max_r_patients)
std_beta_p_r=np.std(beta_max_r_patients)

# means comparison
t1, p1= scipy.stats.ttest_ind(beta_max_patients, beta_max)
t2, p2= scipy.stats.ttest_ind(beta_max_l_patients, beta_max_l)
t3, p3= scipy.stats.ttest_ind(beta_max_r_patients, beta_max_r)

print('Patients, all beta: mean=%s, std=%s' % (mean_beta_p, std_beta_p))
print('Controls, all beta: mean=%s, std=%s' % (mean_beta_c, std_beta_c))
print('ind. sample t-test, all beta: t=%s, p=%s' % (t1, p1))

print('Patients, left beta: mean=%s, std=%s' % (mean_beta_p_l, std_beta_p_l))
print('Controls, left beta: mean=%s, std=%s' % (mean_beta_c_l, std_beta_c_l))
print('ind. sample t-test, left beta: t=%s, p=%s' % (t2, p2))

print('Patients, right beta: mean=%s, std=%s' % (mean_beta_p_r, std_beta_p_r))
print('Controls, right beta: mean=%s, std=%s' % (mean_beta_c_r, std_beta_c_r))
print('ind. sample t-test, right beta: t=%s, p=%s' % (t3, p3))



