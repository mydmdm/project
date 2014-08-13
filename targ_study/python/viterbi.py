# -*- coding: utf-8 -*-
"""
Created on Thu Jul 03 23:16:14 2014

@author: Justin
"""
from __future__ import division
import sys
import math
import itertools
import numpy
import scipy.signal as signal


def int2base_list(idx, base, length_o, polr=0, lsbFirst=0):
    m = []
    length = length_o
    n = base ** (length - 1)
    if idx > n:
        length = int(math.ceil(math.log(idx) / math.log(base)))
        n = base ** (length - 1)
    assert idx >= 0, 'only support non-negative number now'
    for k in range(length):
        x = idx // n
        if polr:
            x = x * 2 - base + 1
        m.append(int(x))
        idx = math.fmod(idx, n)
        n = n // base
    if lsbFirst:
        m.reverse()
    return m[0:length_o]

class class_viterbi:
    '''
    member:
        Q: number of msg values (default binary)
        numSt: number of state
        depth: numSt = Q ** depth
        ideal: ideal values for each (state, msg), len(ideal) = Q ^ (depth + 1)
    '''

    def __init__(self, orderMakov, Q=2, lookAhead=12):
        '''orderMakov is the length of target'''
        assert orderMakov >= 1, 'order of Makov should be positive integer'
        self.Q = Q
        self.depth = orderMakov - 1
        self.numSt = self.Q ** self.depth
        self.state = numpy.arange(self.numSt)
        self.metricSt = numpy.zeros(self.numSt)
        self.trackBack = numpy.zeros((self.numSt, lookAhead))
        self.pt = 0
        self.yNum = 1
        
    def getLatency(self):
        return (self.trackBack.shape[1])

    def stateTransition(self, stCurrent, msg):
        idx = stCurrent * self.Q + msg
        stNext = math.fmod(idx, self.Q ** self.depth)
        return (stNext, idx)

    def filterState(self, stCurrent, msg, target):
        _, idx = self.stateTransition(stCurrent, msg)
        m = int2base_list(idx, self.Q, len(target), polr=1, lsbFirst=1)
#        m = [
#            ((idx // (self.Q ** x)) % self.Q) * self.Q - self.Q / 2
#            for x in range(len(target))
#            ]
#        m.reverse()
#        print idx, m
        s = numpy.dot(target, m)
        return s

    def genIdeal(self, target):
        assert len(target) <= self.depth + 1, 'target too long'
        self.target = target
        self.ideal = [
            self.filterState(st, msg, target)
            for st in range(self.numSt) for msg in range(self.Q)
            ]

    def fetchIdeal(self, idealVal):
        assert len(idealVal) == self.Q ** (self.depth + 1), 'length of ideal values is not acceptable'
        self.ideal = idealVal

    def decision(self):
        idx = numpy.argmin(self.metricSt)
        return int(self.trackBack[idx, self.pt])
        
    def decisionMajority(self):
        num = [
            numpy.count_nonzero(self.trackBack[:, self.pt] == k)
            for k in range(self.Q)
            ]
        return numpy.argmax(num)

    def branchMetric(self, metric, trackBack, Y, stCurrent, msg):
        stNext, idx = self.stateTransition(stCurrent, msg)
        metricNext = metric[stCurrent] + math.pow(Y - self.ideal[idx], 2)
        return metricNext        
    
    def updateState(self, metric, trackBack, Y, stCurrent, msg):
        stNext, idx = self.stateTransition(stCurrent, msg)
        metricNext = self.branchMetric(metric, trackBack, Y, stCurrent, msg)
        #print '%g(%g) ->%g-> %g(%g)\n' % (stCurrent, metric[stCurrent], msg, stNext, metricNext)
        if metricNext <= self.metricSt[stNext]:
            self.metricSt[stNext] = metricNext
            self.trackBack[stNext, :] = trackBack[stCurrent, :]
            self.trackBack[stNext, self.pt] = msg
        
    def detectIterative(self, Y):
        # hd = self.decision()
        hd = self.decisionMajority()
        metric = self.metricSt.copy()
        trackBack = self.trackBack.copy()
        self.metricSt = numpy.ones(self.numSt) * sys.maxint
        [
            self.updateState(metric, trackBack, Y, st, msg)
            for st in range(self.numSt) for msg in range(self.Q)
        ]
        self.pt = math.fmod(self.pt + 1, self.trackBack.shape[1])
        return hd
        
    def detectBatch(self, Y):
        hd = numpy.zeros(Y.shape, dtype='int')
        for k in range(numpy.product(Y.shape)):
            hd.flat[k] = self.detectIterative(
                [Y.flat[x] for x in range(k, k - self.yNum, -1)]
                )
            self.time = k
        return hd
        
 
def test_viterbi(N=1e3):
    target = numpy.array([1,-1])
    bit = numpy.random.randint(2,size=(N, 1))
    data = bit * 2 - 1
    y = signal.lfilter(target, [1.0], data, axis=0)
    y += numpy.random.randn(*y.shape) * math.sqrt(0.2)
    lat = 12
    vit = viterbi(len(target), lookAhead=lat)
    vit.genIdeal(target)
    bitEst = vit.detectBatch(y[lat:])
    errNum = numpy.sum(numpy.bitwise_xor(bit[lat:-lat], bitEst[lat:]))
    print errNum
    
