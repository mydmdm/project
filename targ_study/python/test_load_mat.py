# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 13:30:31 2014

@author: yyang
"""
from __future__ import division

import os
import sys
import math
import itertools as itt

import numpy
import scipy.io as sio
import scipy.signal as signal
import matplotlib.pyplot as plt

import spyder_ctf as ctf
import functions as fun
import viterbi
import noise_predict


def load_waveform(offset_i, read_i):
    '''offset_i: is the index of read offset
    read_i is the index of re-reads (from 0)'''
    wfm_dir = r'../TDK'
    filename = "waveform_TDK_offt%d_rsmp.mat" % (offset_i)
    filename = wfm_dir + r'/' + filename
    print filename
    data = sio.loadmat(filename)
    #
    (B, A, OSR) = ctf.spyder_afe()
    tgt_sel = data['tgt_sel'] - 1
    NRZ = numpy.squeeze(data['nrz_a'][:, tgt_sel])
    #NRZ = NRZ.astype(float).reshape(-1, 1)
    ADC = signal.lfilter(B, A, data['YY_5xsmp_a'][:, read_i], axis=0)
    #if numpy.array(read_i).size == 1:
    #    ADC = ADC.reshape((numpy.size(ADC), 1))
    return (ADC, NRZ, OSR)


def align_adc_w_nrz(ADC, NRZ, OSR,
                    nrz_start=1e3, corr_len=250):
    '''to align and downsampling of ADC by NRZ'''
    ADC = ADC.reshape(-1, OSR, order='C')
    r = int(nrz_start) + numpy.arange(0, corr_len, dtype='int').reshape(-1, 1)
    delay = numpy.zeros(OSR, dtype='int')
    delay_corr = numpy.zeros(OSR)
    delay_corr_full = numpy.zeros((len(r), OSR))
    for i in numpy.arange(0, OSR, dtype='int'):
        delay[i], delay_corr[i] = fun.find_corr_shift(NRZ[r, 0], ADC[r, i])
        delay_corr_full[:, i] = signal.correlate(
            NRZ[r, 0], ADC[r, i], mode='same').ravel()
    delay_corr_full = delay_corr_full.reshape((-1, 1))
    t = fun.range_len_central_zero(
        numpy.size(delay_corr_full)
        ).reshape(-1, 1) / OSR
#    figure(2)
#    clf()
#    plot (t,delay_corr_full[:,0])
    m = numpy.argmax(delay_corr)
    d = delay[m]
#    print d
    X = numpy.roll(ADC[:, m], d)
    X = X.reshape((1, numpy.size(X)))
    return X


def align_adc_w_nrz_upsample(ADC, NRZ, OSR,
                             nrz_start=1e3, corr_len=250):
    #NRZ_os = numpy.zeros((NRZ.shape[0], OSR))
    #NRZ_os[:, 0] = NRZ[:, 0].ravel()
    #NRZ_os = NRZ_os.reshape(-1, 1)
    NRZ_os = numpy.zeros(NRZ.shape[0] * OSR)
    NRZ_os[range(0, len(NRZ_os), OSR)] = NRZ
#    r = int(nrz_start * OSR) + \
#        numpy.arange(0, corr_len * OSR, dtype='int').reshape(-1, 1)
    r = range(int(nrz_start * OSR), int(nrz_start * OSR + corr_len * OSR))
    d, d_max = fun.find_corr_shift(NRZ_os[r], ADC[r])
    ADC = numpy.roll(ADC, d)
#    delay_corr = signal.correlate(NRZ_os[r], ADC[r], mode='same')
#    t = fun.range_len_central_zero(delay_corr.size).reshape(-1, 1) / OSR
#    X = ADC[
#        [int(i * OSR) for i in range(0, int(ADC.size / OSR))], 0
#        ].reshape(-1, 1)
    X = ADC[range(0, len(ADC), OSR)]
    return (X)


def equalization_wiener(NRZ, ADC, PR_TRG,
                        len_eq=16, zero_tap=7,
                        zero_tap_trg=1, nrz_start=1e3, corr_len=1e3):
    '''zero_tap means that delay of W is -zero_tap + [0..len_eq-1]'''
    if rank(ADC) == 1:
        N = 1
    else:
        N = ADC.shape[1]
    len_x = ADC.shape[0]
    len_z = NRZ.shape[0]
    assert len_x == len_z, \
        'NRZ and ADC should be of the same length %d %d' % (len_x, len_z)
    len_trg = numpy.size(PR_TRG)
    d = -int(zero_tap) + numpy.arange(0, len_eq, dtype='int')
    dt = -int(zero_tap_trg) + numpy.arange(0, len_trg, dtype='int')
    r = int(nrz_start) + numpy.arange(0, corr_len, dtype='int')
    #
    A = numpy.zeros((N*len_eq, N*len_eq))
    B = numpy.zeros((N*len_eq, 1))
    for a in numpy.arange(0, N, dtype='int'):
        for b in numpy.arange(0, len_eq, dtype='int'):
            for q in numpy.arange(0, len_trg, dtype='int'):
                B[a * len_eq+b, 0] += numpy.dot(NRZ[r-dt[q]], ADC[r-d[b], a]) * PR_TRG[q]
            for p in numpy.arange(0, N, dtype='int'):
                for q in numpy.arange(0, len_eq, dtype='int'):
                    A[a * len_eq + b, p * len_eq + q] = \
                        numpy.dot(ADC[r - d[q], p], ADC[r - d[b], a])
    W = numpy.dot(numpy.linalg.pinv(A), B).reshape(-1, N, order='F')
    #W = numpy.flipud(W)
    #
    Yid = signal.lfilter(PR_TRG, [1.0], NRZ[:]).reshape(-1, 1)
    Yid = numpy.roll(Yid, -zero_tap_trg)
    Y = numpy.zeros(NRZ.shape).reshape(-1, 1)
#    print NRZ.shape
#    print ADC.shape
#    print Y.shape
#    print W.shape
    for i in range(0, N):
        Y += signal.lfilter(W[:, i], numpy.array([1]), ADC[:, i]).reshape(-1, 1)
    Y = numpy.roll(Y, -zero_tap)
    return (Y, W, Yid, A, B)


def getXcorr(Z, X, type_a, indx_a, type_b, indx_b, dly, start=1e3, clen=1e3):
    '''get cross correlation coeff'''
#    print 'R%s%d%s%d(%d)' % (type_a, indx_a, type_b, indx_b, dly),
    if type_a == 'x':
        aa = X[:, indx_a]
    else:
        aa = Z
    if type_b == 'x':
        bb = X[:, indx_b]
    else:
        bb = Z
    start = int(start)
    clen = int(clen)
    ra = range(start, start + clen)
    rb = range(start + dly, start + clen + dly)
    return numpy.dot(aa[ra], bb[rb])


def equalization_wiener_filter_and_trg(Z, X, PR,
                                       len_dfir=16, len_targ=3):
    if len_targ == 0:
        len_targ = len(PR)
        targ_adpt = 0
        targ_0tap = numpy.argmax(PR)
    else:
        targ_adpt = 1
        targ_0tap = int(len_targ / 2)
        targ_max = max(PR)
    if numpy.rank(X) == 1:
        X = X.reshape(-1, 1)
    N = X.shape[1]
    dfir_dly = range(int(len_dfir / 2) - len_dfir + 1, int(len_dfir / 2) + 1)
    targ_dly = range(-targ_0tap, -targ_0tap + len_targ)
    ALEN = N * len_dfir + len_targ
    A = numpy.zeros((ALEN, ALEN))

#    print dfir_dly
    for a in range(N):
        for b in range(len_dfir):
            type_a = ['x' for p in range(ALEN)]
            indx_a = [a for p in range(ALEN)]
            type_b = ['x' for p in range(N * len_dfir)] \
                + ['b' for p in range(len_targ)]
            indx_b = [p[0] for p in itt.product(range(N), range(len_dfir))] \
                + [0 for p in range(len_targ)]
            sign_r = [1 for p in range(N * len_dfir)] \
                + [-1 for p in range(len_targ)]
            dly = [
                dfir_dly[b] - dfir_dly[p[1]]
                for p in itt.product(range(N), range(len_dfir))
                ] \
                + [dfir_dly[b] - targ_dly[p] for p in range(len_targ)]
            for k in range(ALEN):
                A[a * len_dfir + b, k] = sign_r[k] * getXcorr(Z, X, \
                    type_a[k], indx_a[k], type_b[k], indx_b[k], dly[k])
#            print '\n'
    for b in range(len_targ):
        type_a = ['b' for p in range(ALEN)]
        indx_a = [0 for p in range(ALEN)]
        type_b = ['x' for p in range(N * len_dfir)] \
            + ['b' for p in range(len_targ)]
        indx_b = [p[0] for p in itt.product(range(N), range(len_dfir))] \
            + [0 for p in range(len_targ)]
        sign_r = [-1 for p in range(N * len_dfir)] \
            + [1 for p in range(len_targ)]
        dly = [
            targ_dly[b] - dfir_dly[p[1]]
            for p in itt.product(range(N), range(len_dfir))
            ] \
            + [targ_dly[b] - targ_dly[p] for p in range(len_targ)]
        for k in range(ALEN):
            A[N * len_dfir + b, k] = sign_r[k] * getXcorr(Z, X,
                type_a[k], indx_a[k], type_b[k], indx_b[k], dly[k])
#        print '\n'
    if targ_adpt == 0:
        B = -numpy.dot(A[:, range(N * len_dfir, ALEN)], PR.reshape(-1,1))
        A = numpy.delete(A, obj=range(N * len_dfir, ALEN), axis=1)
        W = numpy.dot(numpy.linalg.pinv(A), B)
        PR_out = PR
        W = numpy.squeeze(W)
    else:
        B = -A[:, N * len_dfir + targ_0tap] * targ_max
        A = numpy.delete(A, obj=N * len_dfir + targ_0tap, axis=1)
        W = numpy.dot(numpy.linalg.pinv(A), B)
        PR_out = numpy.insert(W[N * len_dfir:], targ_0tap, targ_max);
        W = numpy.squeeze(W[range(N * len_dfir)])
        print PR_out
    Y = numpy.zeros(Z.shape)
    for k in range(N):
        Y += signal.lfilter(W[range(k * len_dfir, (k + 1) * len_dfir)], \
            [1.0], X[:, k], axis=0)
    Y = numpy.roll(Y, int(len_dfir / 2) + 1 - len_dfir + numpy.argmax(PR_out))
    Yid = signal.lfilter(PR_out, [1.0], Z, axis=0)
    return (Y, Yid, PR_out, W)


def count_ber(B, D, start=-1, stop=-1, pad=-1):
    length = min(len(B), len(D))
    if pad > 0:
        start = pad
        stop = length - pad
    err = B[start: stop] - D[start: stop]
    numErr = numpy.count_nonzero(err)
    ber = numErr / len(err)
    return (ber, numErr)


def write_fixed_array(A, fxpt=1, Amax=-1, wordLen=6, fname='', fdir='.'):
    if Amax < 0:
        Amax = abs(Amax)
    else:
        Amax = numpy.percentile(A, Amax)
    fx = math.pow(2, wordLen - 1)
    fxMax = fx - 1
    fxMin = -fx
    if fxpt:
        A = numpy.round(A / Amax * fx)
        A[A>fxMax] = fxMax
        A[A<fxMin] = fxMin
    if fname:
        if not os.path.isdir(fdir):
            os.mkdir(fdir)
        numpy.savetxt("%s/%s" % (fdir, fname), A, fmt='%g')
    return A

def prepare_data_file(offset_idx, tdmr=0, read=[0],
                      xave=-1, yave=-1, targ_adpt=0,
                      PR=[8, 14], avg='none',
                      prokey='normal', wdir='work', dump=0,
                      npLen=0,
                      enclen=37728,
                      prml_in=400, prml_out=128, sm_len=14, pad_len=200):
    '''to save file'''
    PR = numpy.array(PR)
    ADC, NRZ, OSR = load_waveform(offset_idx, read)
    X = numpy.zeros((len(NRZ), len(read)))
    for k in range(len(read)):
        X[:, k] = align_adc_w_nrz_upsample(ADC[:, k], NRZ, OSR)
#    print X.shape
#    Y, W, Yid, A, B = equalization_wiener(NRZ, X, PR)
    Y, Yid, PR, W = equalization_wiener_filter_and_trg(
        NRZ, X, PR, len_targ=targ_adpt)
    NRZ_BIT = (1 + NRZ) / 2
    NRZ_BIT = NRZ_BIT.astype('int')
    if npLen == 0:
        vit = viterbi.class_viterbi(len(PR), lookAhead=40)
        vit.genIdeal(PR)
    elif npLen > 0:
        npcal = noise_predict.class_noise_predict(npLen)
        npcal.calibrate(Y, Yid, NRZ_BIT)
        vit = noise_predict.class_npml_det(len(PR), npcal.npLen)
        vit.setTargetAndNp(PR, npcal.npfir, npcal.npmean)

    Zest = vit.detectBatch(Y)
    Zest = numpy.squeeze(Zest)
    Zest = numpy.roll(Zest, - vit.getLatency(), axis=0)
    print numpy.min(vit.metricSt)
    #
    nrzlen = enclen + prml_out + sm_len + pad_len
    pNrzStart = prml_in - prml_out
    pEncStart = pNrzStart + prml_out + sm_len
    #
#    print NRZ.shape
#    print Zest.shape
    Zest[0 :pEncStart] = NRZ_BIT[0 :pEncStart]
    Zest[pEncStart + enclen :-1] = NRZ_BIT[pEncStart + enclen :-1]
    ber, numErr = count_ber(NRZ_BIT, Zest,
                            start=pEncStart, stop=pEncStart+enclen)
    #
    if dump:
        write_fixed_array(
            NRZ_BIT[pNrzStart: pNrzStart + nrzlen],
            fname='%s_nrz.txt' % (prokey),
            fxpt=0, Amax=95, fdir=wdir
        )
        write_fixed_array(
            X[pNrzStart: pNrzStart + nrzlen],
            fname='%s_x.txt' % (prokey),
            Amax=95, fdir=wdir
        )
        write_fixed_array(
            Y[pNrzStart: pNrzStart + nrzlen],
            fname='%s_y.txt' % (prokey),
            Amax=95, fdir=wdir
        )        

    print math.log10(ber)
    print numpy.sum(numpy.square(Y-Yid)) / numpy.sum(numpy.square(PR))
    return (ber, Zest, Y, Yid, X, NRZ_BIT, W, PR)




