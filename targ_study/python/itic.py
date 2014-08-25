# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 14:27:46 2014

@author: yyang
"""

from __future__ import division

import matplotlib.pyplot as pyplot
import math
import numpy

import viterbi
import test_load_mat
import noise_predict


def itic_core(d0, z0, d1, z1):
    

def itic_eval(offset_cent, offset_adj, 
    read_cent=[0], read_adj=[0], 
    PR=[8,14], targ_adpt=0, npLen=0
    itic='yitic'):
    # for central track
    ber, Z, Y, Yid, X, NRZ, W, PR_out = \
        test_load_mat.prepare_data_file(
            offset_idx=offset_cent, read=read_cent, 
            PR=PR, targ_adpt=targ_adpt, npLen=0
            )
    # for adjcent track
    if type(offset_adj) == type(0):
        offset_adj = list([offset_adj])
    for kadj in range(len(offset_adj)):
        _, _, Yadj, _, Xadj, NRZadj, Wadj, PRadj = \
            test_load_mat.prepare_data_file(
                offset_idx=offset_adj[k], read=read_adj,
                PR=PR, targ_adpt=targ_adpt, npLen=npLen
                )
