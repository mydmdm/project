# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 21:51:31 2014

@author: Justin
"""

from __future__ import division


import numpy
import viterbi
import itertools
import matplotlib.pyplot as pyplot

def analyzeTargetDistanceIdealSeq(PR):
    if PR[-1] ==0:
        PR = PR[0:-1]
    n = len(PR)    
    test = viterbi.class_viterbi(len(PR))
    test.genIdeal(PR)

    A = []
    
    for st in range(test.numSt):
        for msg in itertools.product(range(test.Q), repeat=n):
            idx = [0 for x in range(len(msg))]
            s = st
            for k in range(len(msg)):
                s, idx[k] = test.stateTransition(s, msg[k])
            A.append([test.ideal[int(idx[x])] for x in range(len(idx))])
    B = numpy.array(A)
    
    D = [
        numpy.dot(B[x,:] - B[y,:], B[x,:] - B[y,:])
        for x, y in itertools.combinations(range(B.shape[0]), 2)
        ]
    return (numpy.min(D) / dot(PR,PR))

#for st in range(test.numSt):
#    for msg1 in range(test.Q):
#        a, idx1 = test.stateTransition(st, msg1)
#        a = int(a)
#        idx1 = int(idx1)
#        for msg2 in range(test.Q):
#            b, idx2 = test.stateTransition(a, msg2)
#            b = int(b)
#            idx2 = int(idx2)
#            A.append([test.ideal[idx1], test.ideal[idx2]])

#step0 = 0.5
#ra0 = 14
#start0 = 4
#tx0 = [start0 + step0 * x for x in range(int(ra0/step0))]
#step2 = step0
#ra2 = 4
#start2 = 12
#tx2 = [start2 + step2 * x for x in range(int(ra2/step2))]
#
#V = numpy.zeros((len(tx0), len(tx2)))
#for tk in itertools.product(range(len(tx0)), range(len(tx2))):
#    t = [tx0[tk[0]], tx2[tk[1]]]
#    PR = numpy.array([t[0], t[1]])
#    val = analyzeTargetDistanceIdealSeq(PR)
#    V[tk[0], tk[1]] = val
#
#pyplot.figure(1)
#pyplot.clf()
#pyplot.contourf(tx2, tx0, V, 100)
#pyplot.colorbar()
#pyplot.grid(True)    



step0 = 0.2
ra0 = 6
start0 = 4
tx0 = [start0 + step0 * x for x in range(int(ra0/step0))]
step2 = step0
ra2 = 10
start2 = -2
tx2 = [start2 + step2 * x for x in range(int(ra2/step2))]

V = numpy.zeros((len(tx0), len(tx2)))
for tk in itertools.product(range(len(tx0)), range(len(tx2))):
    t = [tx0[tk[0]], tx2[tk[1]]]
    PR = numpy.array([t[0], 22 - t[0] - t[1], t[1]])
    val = analyzeTargetDistanceIdealSeq(PR)
    V[tk[0], tk[1]] = val

pyplot.figure(2)
pyplot.clf()
pyplot.contourf(tx0, tx2, V.T, 20)
pyplot.colorbar()
pyplot.xlabel('tap[0]')
pyplot.ylabel('tap[2]')
pyplot.title('target = tap[0],14,tap[2]')
pyplot.grid(True)    