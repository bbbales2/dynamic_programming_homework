#%%

import numpy
import itertools
import time


#%%

h = 5
w = 5

pieces = [numpy.array([[0, 1], [1, 1]]).astype('uint8'),
          numpy.array([[0, 1, 1], [1, 1, 0]]).astype('uint8'),
          numpy.array([[1], [1]]).astype('uint8')]

#%%
def memoize2(f):
    """ Memoization decorator for a function taking a single argument """
    class memodict(dict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret
    return memodict().__getitem__

def memoize(f):
    """ Memoization decorator for functions taking one or more arguments. """
    class memodict(dict):
        def __init__(self, f):
            self.f = f
        def __call__(self, *args):
            return self[args]
        def __missing__(self, key):
            ret = self[key] = self.f(*key)
            return ret
    return memodict(f)
#%%
h = 5
w = 5

pieces = [numpy.array([[1, 1], [1, 1]]).astype('uint8'),
          numpy.array([[0, 1, 1], [1, 1, 0]]).astype('uint8'),
          numpy.array([[1, 1, 0], [0, 1, 1]]).astype('uint8'),
         numpy.array([[1], [1], [1], [1]]).astype('uint8'),
         numpy.array([[0, 1, 0], [1, 1, 1]]).astype('uint8'),
         numpy.array([[0, 0, 1], [1, 1, 1]]).astype('uint8'),
         numpy.array([[1, 0, 0], [1, 1, 1]]).astype('uint8')]

#%%

@memoize2
def getNextRealState(args):
    state, p, r, o = args
    piece = pieces[p]

    estate = numpy.zeros((4 + h, w)).astype('uint8')
    estate[: h] = numpy.array(state).reshape(h, w)

    p = piece

    if r > 0:
        p = numpy.fliplr(p).transpose()

    if r > 1:
        p = numpy.fliplr(p).transpose()

    if r > 2:
        p = numpy.fliplr(p).transpose()

    #print p

    if o <= w - p.shape[1]:
        for d in range(4 + h - p.shape[0], -1, -1):
            if (estate[d : d + p.shape[0], o : o + p.shape[1]] * p).flatten().sum() != 0:
                break

            lastd = d

        d = lastd
        newState = numpy.array(estate).astype('uint8')
        newState[d : d + p.shape[0], o : o + p.shape[1]] += p

        reward = 0
        removeRows = []

        for k in range(0, estate.shape[0]):
            if newState[k, :].sum() == w:
                removeRows.append(k)
                reward += -1

        removeRows = sorted(removeRows, reverse = True)

        for row in removeRows:
            for row2 in range(row + 1, newState.shape[0]):
                newState[row2 - 1] = newState[row2]
                newState[row2] = 0

        for i in range(h, 4 + h):
            if newState[i].sum() != 0:
                return None, reward

        return tuple(newState[:h].flatten()), reward
    else:
        return None, None

print numpy.array(state).reshape((h, w))
print numpy.array(getNextRealState(state, p, rr, o)[0]).reshape((h, w))

#getNextState((1, 2, 3, 3), 0, 3, 1)

#%%
import bisect

vs = []

ft = 0.0
@memoize2
def features(args):
    state, p = args
    #state = args
    morestate = []
    morestate2 = []
    morestate3 = []
    morestate4 = []

    for i in range(len(state) - 1):
        morestate.append(state[i] | state[i + 1])
        morestate.append(state[i] ^ state[i + 1])
        morestate.append(state[i] & state[i + 1])

    #for i in range(len(morestate) - 1):
    #    morestate2.append(morestate[i] | morestate[i + 1])
    #    morestate2.append(morestate[i] ^ morestate[i + 1])
    #    morestate2.append(morestate[i] & morestate[i + 1])

    #for i in range(len(morestate2) - 1):
    #    morestate3.append(morestate2[i] | morestate2[i + 1])
    #    morestate3.append(morestate2[i] ^ morestate2[i + 1])
    #    morestate3.append(morestate2[i] & morestate2[i + 1])

    state = numpy.array(state)

    state = numpy.concatenate((state, morestate))#, morestate2, morestate3

    #state = numpy.concatenate((state))#, 1 - state

    #nstate = [[0] * len(state)] * len(pieces)
    #nstate[p] = state

    out = state#numpy.concatenate(nstate)


    #holes = 0.0
    #for j in range(w):
    #    for hi in hs:
    #        for i in range(hi):
    #            if state[i, j] == 0.0:
     #               holes += 1

    #features.append(holes)
    piece = numpy.zeros(len(pieces))
    piece[p] = 1.0

    out = numpy.concatenate((state, piece))

    return out

if False:
    heights = numpy.zeros((h, w))

    state = numpy.array(state).reshape(h, w)

    hs = []

    for j in range(w):
        i = 0
        while i <= h:
            if i == h:
                hs.append(h)
                break
            if state[i, j] == 0:
                hs.append(i)
                break
            #print i
            i += 1

    dhs = []
    for i in range(len(hs) - 1):
        dhs.append(hs[i + 1] - hs[i])

    features = []
    for hi in hs:
        v = numpy.zeros(h + 1)
        v[hi] = 1.0
        features.extend(v)

    for hi in dhs:
        v = numpy.zeros(2 * h + 1)
        v[hi + h] = 1.0
        features.extend(v)
    #return out

N = 200

P = numpy.ones((len(pieces), len(pieces))) / float(len(pieces))
rv = numpy.random.random(len(features((tuple([0] * w * h), p)))) * 0.0#01

tmp = time.time()
il = 0.0
sel = 0.0
errors = []
run_lengths = []
rewards = []
T = 0.25
odist = numpy.ones(len(pieces)) / len(pieces)

for t in range(10001):
    state, p = (tuple([0] * w * h), numpy.random.randint(0, len(pieces)))#
    score = 0.0

    Qcfs = []
    Qcs = []
    fvs = []

    z = None

    for i in range(N):
        f = numpy.array(features((state, p)))

        Qc = rv.dot(f)

        tmp1 = time.time()
        nextJs = []
        for r in range(4):
            for o in range(w):
                nextState, reward = getNextRealState((state, p, r, o))

                if reward is not None:
                    total = reward

                    if nextState is not None:
                        for np in range(len(pieces)):
                            total += 0.9 * rv.dot(features((nextState, np))) / len(pieces)

                    nextJs.append(((r, o), total))

        il += time.time() - tmp1

        tmp1 = time.time()

        if t > 50:
            (r, o), Jp = sorted(nextJs, key = lambda x : x[1])[0]
        else:
        #if True:
            Js = numpy.array([Jp for uopt, Jp in nextJs])
            Js /= max(numpy.abs(Js))

            temperature = T * numpy.power(0.99, t) * 1.0 / (1 + numpy.exp(-i + 2)) + 0.1

            exps = numpy.exp(-Js / temperature)

            probs = exps / exps.sum()

            r = numpy.random.random()

            uopt, Jp = nextJs[bisect.bisect_left(numpy.cumsum(probs), r)]

            r, o = uopt

            #print uopt

        nextState, reward = getNextRealState((state, p, r, o))

        r = numpy.random.random()# * odist.sum() -- this sum is always 1.0

        np = bisect.bisect_left(numpy.cumsum(odist), r)

        #print np

        if nextState is not None:
            Qcf = rv.dot(features((nextState, np)))
        else:
            Qcf = 0.0

        score += reward

        dt = reward + 0.9 * Qcf - Qc

        #Qcfs.append(reward + 1.0 * Qcf)
        #Qcs.append(Qc)
        #fvs.append(f)

        if z is not None:
            z = 0.75 * z + f / f.dot(f)
        else:
            z = f / f.dot(f)

        dr = 0.05 * dt * z / numpy.linalg.norm(z)#
        rv = rv + dr
        errors.append(numpy.linalg.norm(dr))

        if nextState is None:
            #print z
            #print 1.0 / (1.0 + numpy.exp(-i + 2.5 + t / 1000.0))
            break

        state = nextState
        p = np
        sel += time.time() - tmp1
        #1/0
    #for Qcf, Qc, f in zip(Qcfs, Qcs, fvs):

    #1/0
    #print t
    run_lengths.append(i)
    rewards.append(score)

    if t % 50 == 0:
        print t, numpy.mean(run_lengths[-50:]), numpy.mean(errors[-50:]), numpy.mean(rewards[-50:]), temperature

    #1/0
        #print rv

    #Js[i][(state, p)] = Jp
    #us[i][(state, p)] = uopt
print "Total: ", time.time() - tmp
print "Features: ", ft
print "Inner loop: ", il
print "Selection time: ", sel
#%%

            #board2state(numpy.array(nextState).reshape(w, h))


    r, o = us[i][(, p)]
    vs.append(total)

#%%
import bisect

vs = []

for rpts in range(100):
    state, p = (tuple([0] * w * h), numpy.random.randint(len(pieces)))
    total = 0.0

    #print 'HI', p
    for i in range(N):
        f = numpy.array(features((state, p)))

        Qc = rv.dot(f)

        nextJs = []
        for r in range(4):
            for o in range(w):
                nextState, reward = getNextRealState((state, p, r, o))

                if reward is not None:
                    t = reward

                    if nextState is not None:
                        for np in range(len(pieces)):
                            t += 1.0 * rv.dot(features((nextState, np))) / len(pieces)

                    nextJs.append(((r, o), t))

        (r, o), Jp = sorted(nextJs, key = lambda x : x[1])[0]

        print r, o, nextJs
        1/0

        nextState, reward = getNextRealState((state, p, r, o))

        if reward != None:
            total += reward
            if nextState != None:

                state = nextState

                r = numpy.random.random() * sum(odist)

                p = bisect.bisect_left(numpy.cumsum(odist), r)
            else:
                break

    print total
    vs.append(total)

print numpy.mean(vs), numpy.std(vs)
#%%

state = ((0, 0, 0, 0, 0, 0, 0, 0, 0), 1)

for stage in range(0, N):
    c2g, nstate = paths[stage][state]
    print c2g
    print state, stage % 3, nstate
    print pieces[state[1]]
    print numpy.array(nstate[0]).reshape(h, w)
    print '----'
    state = nstate
#%%