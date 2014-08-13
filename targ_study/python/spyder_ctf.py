# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 23:31:19 2014

@author: Justin
"""
from __future__ import division
import numpy as np
import math as math
import scipy.signal as signal

def db2lin (db,dec=20):
    lin = math.pow(10,db/dec)
    return lin


def all_pole_flt_2nd_order (q,zeta,Rc,Rs):
    '''Rs: sampling rate
    Rc: cut-off frequency'''
    a = 2 * math.pi * zeta * (Rc / Rs)
    des = 4 + 2/q * a + a**2
    c0 = (8 - 2* a**2) / des
    c1 = (-4 + 2/q * a - a**2) / des
    g  = (1-c0-c1) / 4
    N = np.array([1,2,1])
    D = np.array([1,-c0,-c1])
    return (N,D,g)


def real_zero_flt_1st_order (zeta,polr,zero_loc,Rc,Rs):
    '''Rs: sampling rate
    Rc: cut-off frequency'''
    a = 2 * polr * math.pi * zeta * (Rc / Rs) * zero_loc
    r0 = (2 - a) / (2 + a)
    r0[r0==1] = 0.0
    g = 2 / (1 - r0)
    N = np.hstack((np.ones(r0.shape),-r0))
    N = np.expand_dims(N,axis=0)
    D = np.array([1.0])
    return (N,D,g)


def spyder_ctr_mar13 (LX,CTF_Freq,CTF_Boost,BUFGain):
    '''% The 6th order Butterworth filter has the following sections:
    %   (i)     Second-order all-pole stage with q = 1,
    %   (ii)    Second-order all-pole stage with q = 1,
    %   (iii)   Second-order all-pole stage with q = 0.5,
    %   (iv)   First-order real zero of LHP,
    %   (v)    Matched first-order real zero in RHP.
    % Below, the discrete equivalent of each of these stages is derived.'''
    Data_Rate = 100e6
    zeta = 1.0
    osr = LX
    sr = osr * Data_Rate
    fs = CTF_Freq * Data_Rate
    if CTF_Boost > 12:
        DCGain = BUFGain - (CTF_Boost-12)
    else:
        DCGain = BUFGain
    ctfgain = db2lin(DCGain)
    boost = CTF_Boost
    #
    len_b = len(boost)
    zero_loc = np.zeros(len(boost)) 
    zero_loc
    for i in range(0,len_b):
        if (boost[i] != 0):
            (math.sqrt(db2lin(boost[i]))-1)
            zero_loc[i] = 1/math.sqrt(db2lin(boost[i])-1)/zeta
    len_z = len_b
    # (i)
    (N1,D1,g) = all_pole_flt_2nd_order (1.0,1.1554,fs,sr)
    ctfgain *= g
    # (ii)
    (N2,D2,g) = all_pole_flt_2nd_order (1.0,1.1554,fs,sr)
    ctfgain *= g
    # (iii)
    (N3,D3,g) = all_pole_flt_2nd_order (0.5,1.1554,fs,sr)
    N3 = np.array([1.0])
    ctfgain *= g    
    #(iv)
    (N4,D4,g) = real_zero_flt_1st_order(1.0,1,zero_loc,fs,sr)
    ctfgain *= g
    #(v)
    (N5,D5,g) = real_zero_flt_1st_order(1.0,-1,zero_loc,fs,sr)
    ctfgain *= g
    #
    des = D1
    des = np.convolve(des,D2)
    des = np.convolve(des,D3)
    des = np.convolve(des,D4)
    des = np.convolve(des,D5)
    num = N1
    num = np.convolve(num,N2)
    num = np.convolve(num,N3)
    num1 = np.zeros((len_z,len(num)+len(N4[0,:])+len(N5[0,:])-2))
    for i in range(0,len_z):
        num1[i,:] = ctfgain[i] * np.expand_dims(np.convolve(num,np.convolve(N4[i,:],N5[i,:])),axis=0)
    Filt_B = num1
    Filt_A = des
#    print zero_loc
#    print 'fs = ',fs
#    print 'sr = ',sr
#    print 'D1 = ',D1
#    print 'D2 = ',D2
#    print 'D3 = ',D3
#    print 'N4 = ',N4
#    print 'N5 = ',N5
    return (Filt_B,Filt_A)
    
def spyder_afe ():
    Filt_ON = [1,1]
    LX = 5
    ACC_Freq = 0.1
    CTF_Freq = 37
    CTF_Boost=np.array([7])
    BUFGain = 4.5
    if Filt_ON[0]==1:
        (B_ACC,A_ACC) = signal.butter(1,ACC_Freq*0.02/LX,'high')
    else:
        B_ACC = np.array([1.0,0.0,0.0])
        A_ACC = np.array([1.0,0.0,0.0])
    if Filt_ON[1]==1:
        (B_CTF,A_CTF) = spyder_ctr_mar13 (LX,CTF_Freq*0.01,CTF_Boost,BUFGain)
    else:
        B_CTF = np.array([1.0,0.0,0.0])
        A_CTF = np.array([1.0,0.0,0.0])  
    B_AFE = np.convolve(B_ACC,B_CTF[0,:])
    A_AFE = np.convolve(A_ACC,A_CTF)
    return (B_AFE,A_AFE,LX)