# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 22:41:55 2014

@author: Justin
"""

import math
import numpy as numpy
import viterbi
import matplotlib.pyplot as plt
import scipy.signal as signal

def map_bitSeq_npIndex(bit, npLen, base=2):
    '''LEFT is latest bit'''
    idx = int(
        ''.join(
            [str(bit[x]) for x in range(len(bit))]
            ),
        base=base)
    return idx

def noise_prediction_error_pattern(BIT, EST, Y, Yid, npLen=5, npLA=0):
    '''npLA: numpy look ahead'''
    npidx = signal.lfilter(
        numpy.power(2, numpy.arange(npLen)), [1.0], BIT, 
        axis=0)
    Ye = Y - Yid
    errPat = [[] for k in range(2**npLen)]
    errSmp = [[] for k in range(2**npLen)]
    ra = numpy.nonzero(BIT - EST)
    for k in range(len(ra)):
        idx = int(npidx[ra[k] + npLA])
        errPat[idx].append(
            [BIT[x] for x in range(k + npLA, k + npLA - npLen, -1)]
        )
        errsmp[idx].append(
            [Ye[x] for x in range(k + npLA, k + npLA - npLen, -1)]
        )
    
    plt.figure(1)

def noise_prediction_cal(Y, Yid, BIT, npLen=5):
    E = Y - Yid
    npNum = 2 ** npLen
    fout = [[] for k in range(npNum)]
    npfir = [numpy.zeros((npLen - 1, 1)) for k in range(npNum)]
    npmean = numpy.zeros(npNum)
    npidx = numpy.zeros(len(BIT), dtype='int')
    for k in range(npLen, len(BIT)):
#        idx = int(
#            ''.join(
#                [str(BIT[x]) for x in range(k, k - npLen, -1)]
#                ),
#            base=2)
        idx = map_bitSeq_npIndex(
            [BIT[x] for x in range(k, k - npLen, -1)],
            npLen)
        npidx[k] = idx
        fout[idx].append(
            [E[x] for x in range(k, k - npLen, -1)],
            )
    for k in range(npNum):
        Rxx = numpy.zeros((npLen - 1, npLen - 1))
        P = numpy.zeros((npLen - 1, 1))
        S = 0
        for m in range(len(fout[k])):
#            x = numpy.array(
#                [fout[k][m][n] for n in reversed(range(npLen - 1))]
#                ).reshape(-1, 1)
            x = numpy.array(fout[k][m][1:]).reshape(-1, 1)
            Rxx += x * x.T
            P += fout[k][m][0] * x
            S += fout[k][m][0]
        npfir[k] = numpy.vstack((
            1.0, 
            -numpy.dot(numpy.linalg.pinv(Rxx), P)
            )).reshape(-1, 1)
        npmean[k] = S / len(fout[k])
    return (fout, npfir, npmean, npidx)


def noise_prediction_verify(fout, npfir, npmean, npLen=5, 
                            EST=[0], BIT=[0], IDX=[0]):
    npNum = 2 ** npLen
    var_err = [0 for k in range(npNum)]
    var_err_np = [0 for k in range(npNum)]
    for k in range(npNum):
        E = numpy.array(fout[k])
        #NP = numpy.array([npfir[k][x] for x in reversed(range(len(npfir[k])))])
        var_err[k] = numpy.var(E[:, 0])
        var_err_np[k] = numpy.var(E.dot(npfir[k]) - npmean[k])
    ber_per_np = []
    if len(BIT) > 1 and len(BIT) == len(EST) and len(BIT) == len(IDX):
        ber_per_np = [0 for k in range(npNum)]
        for k in range(1000, len(BIT)-1000):
            if EST[k] != BIT[k]:
                ber_per_np[IDX[k+2]] += 1
    plt.figure(1)
    plt.clf()
    plt.subplot(2,1,1)
    plt.hold(True)
    plt.plot(numpy.arange(npNum), var_err, '-bo')
    plt.plot(numpy.arange(npNum), var_err_np, '-ro')
    plt.hold(False)
    plt.grid(True)
    plt.subplot(2,1,2)
    if ber_per_np:
        plt.plot(numpy.arange(npNum), ber_per_np, '-ro')
        plt.grid(True)


class class_noise_predict:
    '''noise prediction by wiener theory'''

    def __init__(self, npLen):
        self.npLen = npLen

    def calibrate(self, Y, Yid, BIT):
        self.fout, self.npfir, self.npmean, self.npidx = \
            noise_prediction_cal(Y, Yid, BIT, self.npLen)

    def verify(self, EST=[0], BIT=[0]):
        noise_prediction_verify(
            self.fout, self.npfir, self.npmean, self.npLen, 
            EST, BIT, self.npidx)


class class_npml_det(viterbi.class_viterbi):
    '''detector with npcal'''
    
    def __init__(self, tarLen, npLen, Q=2, lookAhead=12):
        viterbi.class_viterbi.__init__(self, tarLen + npLen - 1)
        self.tarLen = tarLen
        self.npLen = npLen
        self.yNum = tarLen + npLen - 1

    def setTargetAndNp(self, target, npfir, npmean):
        self.target = target
        self.npfir = npfir
        self.npmean = npmean
        assert self.tarLen == len(target), \
            'target len is set to %d' % (self.tarLen)
        assert self.npLen == len(npfir[0]), \
            'npfir len is set to %d' % (self.npLen)
    
    def branchMetric(self, metric, trackBack, Y, stCurrent, msg):
        _, idx = self.stateTransition(stCurrent, msg)
        s = viterbi.int2base_list(idx, self.Q, self.depth + 1,
                                  polr=0, lsbFirst=1)
        sa = numpy.array(s)
        E = numpy.array([
            Y[k] - (sa[k: k + self.tarLen]*2 - 1).dot(self.target)
            for k in range(0, self.npLen)
            ])
        npIdx = map_bitSeq_npIndex(s[0: self.npLen], self.npLen)
        metricNext = metric[stCurrent] +\
            math.pow(E.dot(self.npfir[npIdx]), 2)
        return metricNext