#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 13:58:47 2020

This script is a repository of manually selected information

It contains, for each subject
A. subject id (e.g. 5815, 'sib24')
B. individual peak frequency left hemisphere (a pair of numbers, e.g. 17, 22 Hz, corresponding to lower and upper limit)
C. peak channel left hemisphere (e.g. 'MEG0222')
D. individual peak frequency right hemisphere
E. peak channel right hemisphere

Empty entries are denoted by NaN and [] 

output is a variable which stores this information (beta_info_f_ch)

"""

# file for storing info on peak beta frequency and peak CMC channels
# channels and frequency information manually selected from output of 
# script CMC_ch_selection.py

def info_beta_freqs_chans():

    beta_info_f_ch=[]
  
    
    # OLD PATIENTS
    a=[3936, 13, 18, ['MEG0422'], 14, 19, ['MEG1112']]
    b=[4033, 'NaN', 'NaN', [], 17, 20, ['MEG1123']]
    c=[4034, 18, 23, ['MEG0222'], 18, 22, ['MEG1313']]
    d=[4114, 15, 19, ['MEG0412'], 17, 21, ['MEG1113']]
    e=[4227, 18, 22, ['MEG0222'], 19, 23, ['MEG1312']]
    f=[4247, 'NaN', 'NaN', [], 14, 19, ['MEG1042']]
    g=[4427, 18, 23, ['MEG0413'], 18, 23, ['MEG1312']] # change in freq. OFF vs. ON
    h=[4429, 19, 23, ['MEG0412'], 20, 24, ['MEG1312']]# change in freq. OFF vs. ON
    i=[4430, 23, 27, ['MEG0422'], 17, 21, ['MEG1123']] 
    j=[4745, 17, 21, ['MEG0222'], 17, 22, ['MEG1123']]
    
    beta_info_f_ch.append(a)
    beta_info_f_ch.append(b)
    beta_info_f_ch.append(c)
    beta_info_f_ch.append(d)
    beta_info_f_ch.append(e)
    beta_info_f_ch.append(f)
    beta_info_f_ch.append(g)
    beta_info_f_ch.append(h)
    beta_info_f_ch.append(i)
    beta_info_f_ch.append(j)    
    
    # NEW PATIENTS
    a=[5815, 17, 22, ['MEG0222'], 16, 21, ['MEG1312']]
    b=[5826, 18, 22, ['MEG0222'],  17, 22, ['MEG1312']]
    c=[5832, 17, 21, ['MEG0422'], 17, 22, ['MEG1123']]
    d=[5874, 17, 21, ['MEG0413'], 16, 20, ['MEG1112']]
    e=[5875, 18, 23, ['MEG0222'], 19, 23, ['MEG1312']]
    f=[5912, 18, 23, ['MEG0222'], 20, 24, ['MEG1112']]
    g=[5926, 21, 26, ['MEG0222'], 'NaN', 'Nan', []]
    h=[5938, 20, 25, ['MEG0422'], 19, 24, ['MEG1112']]
    i=[5952, 20, 24, ['MEG0213'], 17, 23, ['MEG1312']]
    j=[5953, 20, 24, ['MEG0422'], 20, 24, ['MEG1112']]    
    k=[5962, 12, 16, ['MEG0222'], 13, 17, ['MEG1312']]
    l=[5976, 14, 20, ['MEG0422'], 13, 18, ['MEG1112']]
    m=[5987, 16, 22, ['MEG0413'], 17, 23, ['MEG1312']]
    
    beta_info_f_ch.append(a)
    beta_info_f_ch.append(b)
    beta_info_f_ch.append(c)
    beta_info_f_ch.append(d)
    beta_info_f_ch.append(e)
    beta_info_f_ch.append(f)
    beta_info_f_ch.append(g)
    beta_info_f_ch.append(h)
    beta_info_f_ch.append(i)
    beta_info_f_ch.append(j)
    beta_info_f_ch.append(k)
    beta_info_f_ch.append(l)
    beta_info_f_ch.append(m)      
    
    # CONTROLS
    a=['sib24', 15, 21, ['MEG0222'], 23, 28, ['MEG1123']]
    b=['sib26', 13, 18, ['MEG0222'],  15, 20, ['MEG1312']]
    c=['sib85', 17, 22, ['MEG0413'], 19, 25, ['MEG1312']]
    d=['sib124', 15, 20, ['MEG0223'], 16, 22, ['MEG1123']]
    e=['sib152', 14, 21, ['MEG0222'], [],[], [] ]           # old head transform
    f=['sib155', 18, 25, ['MEG0413'], 15, 22, ['MEG1112']]  
    g=['sib156', 14, 18, ['MEG0222'], 14, 18, ['MEG1122']]  # old head transform
    h=['sib212', 16, 25, ['MEG0413'], 16, 26, ['MEG1312']]
    i=['s09', 14, 20, ['MEG0413'], 13, 20, ['MEG1112']]
    j=['s10', 19, 24, ['MEG0222'], 15, 21, ['MEG1123']]
    k=['s11', 18, 25, ['MEG0222'], 18, 26, ['MEG1123']]
    l=['s15', 17, 23, ['MEG0413'], 18, 23, ['MEG1123']]
    m=['s16', 15, 22, ['MEG0413'], 14, 21, ['MEG1312']]
    n=['s17', 15, 20, ['MEG0222'], 14, 20, ['MEG1112']]
    o=['s18', 14, 21, ['MEG0413'], 14, 20, ['MEG1123']]
    p=['s19', 20, 28, ['MEG0222'], 20, 27, ['MEG1312']]
    q=['s20', 23, 28, ['MEG0422'], 23, 28, ['MEG1123']]
    r=['s21', 14, 19, ['MEG0423'], 14, 19, ['MEG1313']]
    s=['s22', 15, 20, ['MEG0423'], 15, 20, ['MEG1123']]
    t=['s23', 14, 19, ['MEG0222'], 16, 22, ['MEG1112']]
    u=['s24', 14, 20, ['MEG0222'], 16, 22, ['MEG1112']]
    v=['s25', 20, 28, ['MEG0422'], 19, 25, ['MEG1112']]
    w=['s26', 16, 24, ['MEG0222'], 17, 23, ['MEG1112']] 
    
    beta_info_f_ch.append(a)
    beta_info_f_ch.append(b)
    beta_info_f_ch.append(c)
    beta_info_f_ch.append(d)
    beta_info_f_ch.append(e)
    beta_info_f_ch.append(f)
    beta_info_f_ch.append(g)
    beta_info_f_ch.append(h)
    beta_info_f_ch.append(i)
    beta_info_f_ch.append(j)
    beta_info_f_ch.append(k)
    beta_info_f_ch.append(l)
    beta_info_f_ch.append(m)
    beta_info_f_ch.append(n)
    beta_info_f_ch.append(o)
    beta_info_f_ch.append(p)
    beta_info_f_ch.append(q)
    beta_info_f_ch.append(r)
    beta_info_f_ch.append(s)
    beta_info_f_ch.append(t)
    beta_info_f_ch.append(u)
    beta_info_f_ch.append(v)
    beta_info_f_ch.append(w)  
    
    
    return beta_info_f_ch
    

