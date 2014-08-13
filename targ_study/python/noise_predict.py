# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 22:41:55 2014

@author: Justin
"""

import math
import numpy as np
import viterbi
import matplotlib.pyplot as plt

def map_bitSeq_npIndex(bit, npLen, base=2):
    '''LEFT is latest bit'''
    idx = int(
        ''.join(
            [str(bit[x]) for x in range(len(bit))]
            ),
        base=base)
    return idx

def noise_prediction_cal(Y, Yid, BIT, npLen=5):
    E = Y - Yid
    npNum = 2 ** npLen
    fout = [[] for k in range(npNum)]
    npfir = [np.zeros((npLen - 1, 1)) for k in range(npNum)]
    npmean = np.zeros(npNum)
    npidx = np.zeros(len(BIT), dtype='int')
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
        Rxx = np.zeros((npLen - 1, npLen - 1))
        P = np.zeros((npLen - 1, 1))
        S = 0
        for m in range(len(fout[k])):
#            x = np.array(
#                [fout[k][m][n] for n in reversed(range(npLen - 1))]
#                ).reshape(-1, 1)
            x = np.array(fout[k][m][1:]).reshape(-1, 1)
            Rxx += x * x.T
            P += fout[k][m][0] * x
            S += fout[k][m][0]
        npfir[k] = np.vstack((
            1.0, 
            -np.dot(np.linalg.pinv(Rxx), P)
            )).reshape(-1, 1)
        npmean[k] = S / len(fout[k])
    return (fout, npfir, npmean, npidx)


def noise_prediction_verify(fout, npfir, npmean, npLen=5, 
                            EST=[0], BIT=[0], IDX=[0]):
    npNum = 2 ** npLen
    var_err = [0 for k in range(npNum)]
    var_err_np = [0 for k in range(npNum)]
    for k in range(npNum):
        E = np.array(fout[k])
        #NP = np.array([npfir[k][x] for x in reversed(range(len(npfir[k])))])
        var_err[k] = np.var(E[:, 0])
        var_err_np[k] = np.var(E.dot(npfir[k]) - npmean[k])
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
    plt.plot(np.arange(npNum), var_err, '-bo')
    plt.plot(np.arange(npNum), var_err_np, '-ro')
    plt.hold(False)
    plt.grid(True)
    plt.subplot(2,1,2)
    if ber_per_np:
        plt.plot(np.arange(npNum), ber_per_np, '-ro')
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
        sa = np.array(s)
        E = np.array([
            Y[k] - (sa[k: k + self.tarLen]*2 - 1).dot(self.target)
            for k in range(0, self.npLen)
            ])
        npIdx = map_bitSeq_npIndex(s[0: self.npLen], self.npLen)
        metricNext = metric[stCurrent] +\
            math.pow(E.dot(self.npfir[npIdx]), 2)
        return metricNext