# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 14:02:45 2014

@author: Justin
"""

import matplotlib.pyplot as plt
import math

import viterbi
import test_load_mat
import noise_predict

ber, Z, Y, Yid, X, NRZ, W, PR_out = test_load_mat.prepare_data_file(
    offset_idx=19, read=[2], 
    PR=[8, 14], targ_adpt=0,
    npLen=0)

#%% basic test
reload(viterbi)
reload(noise_predict)
reload(test_load_mat)

ber, Z, Y, Yid, X, NRZ, W, PR_out = test_load_mat.prepare_data_file(
    offset_idx=19, read=[2], 
    PR=[8, 14], targ_adpt=0,
    npLen=0)


#%% histgram of y-yid
reload(noise_predict)
npcal = noise_predict.class_noise_predict(5)
npcal.calibrate(Y, Yid, NRZ)
npcal.verify(Z, NRZ)

#%% npml
reload(noise_predict)
N = -1
npml = noise_predict.class_npml_det(len(PR_out), npcal.npLen)
npml.setTargetAndNp(PR_out, npcal.npfir, npcal.npmean)
Znp = npml.detectBatch(Y[0:N])
Znp = np.squeeze(Znp)
Znp = np.roll(Znp, - npml.getLatency(), axis=0)
ber2, _ = test_load_mat.count_ber(NRZ, Znp, pad=100)
print math.log10(ber2)

#%% 
figure(1)
plt.clf()
plt.hold(True)
zid = np.zeros(len(fout))
for k in range(len(fout)):
    zid[k] = sum([fout[k][x][4] for x in range(len(fout[k]))]) / len(fout[k])
    _ = plt.hist([fout[k][x][4] for x in range(len(fout[k]))],
                  bins=32, range=[-32,32], histtype='step')
plt.hold(False)
plt.grid(True)

figure(2)
plt.clf()
plt.hold(True)
plt.plot(zid,'-bs')
plt.hold(False)
plt.grid(True)

#%% test for different target
ber = [
    test_load_mat.prepare_data_file(
    offset_idx=19, read=[0], PR=[k, 14], targ_adpt=0)[0]
    for k in range(6,13)
    ]


plt.figure(1)
plt.clf()
plt.hold(True)
#plt.plot(Y[range(1000)], '-bs')
#plt.plot(Yid[range(1000)], '-r.')
plt.plot(range(6,13),ber)
plt.hold(False)
plt.grid(True)

#%% test for different track
reload(viterbi)
reload(noise_predict)
reload(test_load_mat)

track_idx = range(17,22)
targ_adpt_arr = [0, 2]
ber0 = [[] for x in range(len(targ_adpt_arr))]
for t in range(len(targ_adpt_arr)):
    ber0[t] = [
        test_load_mat.prepare_data_file(
            offset_idx=19, read=[0], 
            PR=[8,14], targ_adpt=targ_adpt_arr[t],
            npLen=5
            )[0]
        for k in track_idx
        ]


#%%
plt.figure(1)
plt.clf()
plt.hold(True)
plt.plot(track_idx, np.log10(np.array(ber0[0])), '-bs')
plt.plot(track_idx, np.log10(np.array(ber0[1])), '-rs')
plt.hold(False)
plt.grid(True)

#%%
reload(viterbi)
reload(noise_predict)
reload(test_load_mat)


track_idx = range(19,20)
ber1 = [
    test_load_mat.prepare_data_file(
        offset_idx=19, read=[0], 
        PR=[8,14], targ_adpt=3,
        npLen=5
        )[0]
    for k in track_idx
    ]

#%%
reload(viterbi)
reload(noise_predict)
reload(test_load_mat)


t0_tap = range(8, 14)
ber2 = [
    test_load_mat.prepare_data_file(
        offset_idx=19, read=[0], 
        PR=[k,22-k], targ_adpt=0,
        npLen=0
        )[0]
    for k in t0_tap
    ]
plt.figure(1)
plt.clf()
plt.hold(True)
plt.plot(t0_tap, np.log10(np.array(ber2)), '-bs')
plt.hold(False)
plt.grid(True)

#%%
reload(viterbi)
reload(noise_predict)
reload(test_load_mat)

offset = 19
read_arr = [k for k in range(8)]
ber3 = [test_load_mat.prepare_data_file(
    offset_idx=offset, read=[r], 
    PR=[8, 14], targ_adpt=0, npLen=0,
    dump=0, prokey='tdk_wave',
    wdir='work/offset%d/read%d' % (offset, r),
    )[0]
    for r in read_arr
    ]
plt.figure(1)
plt.clf()
plt.hold(True)
plt.plot(read_arr, np.log10(np.array(ber3)), '-bs')
plt.hold(False)
plt.grid(True)    