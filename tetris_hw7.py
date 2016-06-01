#%%
import os
import pyximport
pyximport.install(reload_support=True)

os.chdir('/home/bbales2/Documents/classes/ece271/')
import tetris as tl
import time
import sys
#%%
reload(tl)


import sklearn.linear_model
import numpy
import itertools
import time

N = 10000

Npieces = len(tl.pieces)

w = tl.w_
h = tl.h_

P = numpy.ones((Npieces, Npieces)) / float(Npieces)
rv = numpy.random.random(len(tl.features((tuple([0] * w * h), 0)))) * 0.1

rv[: w] = 0
rv[w : 2 * w] = 0
rv[-1] = 1
rv[-2] = 10

tmp = time.time()
il = 0.0
sel = 0.0
trn = 0.0
errors = []
run_lengths = []
rewards = []
T = 0.25

lr = sklearn.linear_model.LinearRegression(fit_intercept = False)
rb = 0.0
guess_next = 0.0
recordings = []
for t in range(1001):
    if t > 0 and t % 100 == 0:
        tmp1 = time.time()
        Xs = []
        Ys = []
        for rec in recordings:
            for i in range(len(rec)):
                tsum = rec[i][1]
                for j in range(i, len(rec)):
                    tsum += 0.5**(j - i) * rec[j][2]

                Xs.append(rec[i][0])
                Ys.append(tsum)
        Xs = numpy.array(Xs)
        Ys = numpy.array(Ys)

        lr.fit(Xs, Ys)

        rv = lr.coef_# * (1 - 0.5) + rv * 0.5
        #1/0
        #rn = numpy.linalg.lstsq(Xs, Ys)

        #rv = rn[0]

        print 'recomputed...'

        recordings = []
        trn += time.time() - tmp1

    recording = []

    state, p = (tuple([0] * w * h), numpy.random.randint(0, Npieces))#
    score = 0.0

    for i in range(N):
        f = numpy.array(tl.features((state, p)))
        Qc = rv.dot(f)

        nextJs = []
        height = tl.getHeight(state)

        for r in range(4):
            for o in range(w):
                tmp1 = time.time()
                nextState, reward = tl.getNextRealState((state, p, r, o, height))
                il += time.time() - tmp1

                if reward is not None:
                    total = reward

                    if nextState is not None:
                        #for np in range(len(pieces)):
                        total += rv.dot(tl.features((nextState, 0)))

                        nextJs.append(((r, o), total))

        tmp1 = time.time()
        if len(nextJs) == 0:
            #1/0
            break

        (r, o), Jp = sorted(nextJs, key = lambda x : x[1])[0]

        nextState, reward = tl.getNextRealState((state, p, r, o, tl.getHeight(state)))

        #print (r, o), nextJs
        #print numpy.reshape(nextState, (h, w))
        #print tl.pieces[p][r]
        #print numpy.reshape(state, (h, w))
        #print '----'

        np = numpy.random.randint(0, Npieces)

        score += reward

        if nextState is not None:
            Qcf = rv.dot(tl.features((nextState, np)))
        else:
            Qcf = 0.0
            break

        dt = reward + Qcf - Qc

        recording.append((f, Qc, dt))

        #if nextState is None:
        #    break

        state = nextState
        p = np

        sel += time.time() - tmp1

    run_lengths.append(i)
    rewards.append(score)
    recordings.append(recording)

    if t % 10 == 0:
        print t, numpy.mean(run_lengths[-50:]), numpy.mean(rewards[-50:])
        sys.stdout.flush()

    #1/0
        #print rv

    #Js[i][(state, p)] = Jp
    #us[i][(state, p)] = uopt

print "Total: ", time.time() - tmp
#print "Features: ", ft
print "Inner loop: ", il
print "Selection time: ", sel
print "Train time: ", trn


#tmp = time.time()
#tetris.run(10, 10, 0.1, 0.99, 0.95)
#print "Total: ", time.time() - tmp

#print tetris.gns
