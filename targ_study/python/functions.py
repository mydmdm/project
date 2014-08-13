# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 10:19:06 2014

@author: yyang
"""
from __future__ import division
import numpy as np
import math as math
import scipy.signal as signal

def range_len_central_zero(L):
    t0 = -np.floor(L/2.0)
    t1 = np.floor((L-1)/2.0)
    t = np.arange(t0,t1+1,dtype='int')
    return t

def find_corr_shift (a,b):
    '''d: return value
    a = np.roll(b,d)'''
    t = range_len_central_zero(len(a))
    C = signal.correlate(a,b,mode='same')
    p = np.argmax(C)
    return (t[p],C[p])

def corr_shift (a,b,shift_a=0,shift_b=0):
    '''return y = sum_n(a(n-shift_a) * b(n-shift_b))'''
    c = np.dot(np.roll(a,shift_a),np.roll(b,shift_b))
    c = c / np.size(a,1)
    return c