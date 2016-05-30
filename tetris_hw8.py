#%%

import numpy
import itertools
import time
import collections

#%%

h = 5
w = 5

pieces = [numpy.array([[0, 1], [1, 1]]).astype('uint8'),
          numpy.array([[0, 1, 1], [1, 1, 0]]).astype('uint8'),
          numpy.array([[1], [1]]).astype('uint8')]


def memoize2(f):
    """ Memoization decorator for a function taking a single argument """
    class memodict(dict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret
    return memodict().__getitem__

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

    estate = numpy.zeros((2 * h, w)).astype('uint8')
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
        for d in range(2 * h - p.shape[0], -1, -1):
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

        for i in range(h, 2 * h):
            if newState[i].sum() != 0:
                return None, reward

        return tuple(newState[:h].flatten()), reward
    else:
        return None, None

#%%
import bisect

vs = []

ft = 0.0
@memoize2
def features(args):
    tmp = time.time()
    state, p = args
    morestate = []

    for i in range(len(state) - 1):
        morestate.append(state[i] * state[i + 1])

    state = numpy.array(state).copy() - 0.5

    state = numpy.concatenate((state, morestate))

    nstate = [[0] * len(state)] * len(pieces)
    nstate[p] = state

    out = numpy.concatenate(nstate)

    global ft
    ft += time.time() - tmp
    return out

N = 2000

P = numpy.ones((len(pieces), len(pieces))) / float(len(pieces))
rv = numpy.random.random(len(features((tuple([0] * w * h), numpy.random.randint(0, len(pieces)))))) * 0.0

tmp = time.time()
il = 0.0
sel = 0.0
errors = []
run_lengths = []
T = 1.0

odist = numpy.ones(len(pieces)) / len(pieces)

Q = collections.defaultdict(lambda : 0)

alpha = 0.25

for t in range(1000):
    state, p = (tuple([0] * w * h), numpy.random.randint(0, len(pieces)))#
    total = 0.0

    for i in range(10):
        tmp1 = time.time()
        nextJs = []
        for r in range(4):
            for o in range(w):
                nextState, reward = getNextRealState((state, p, r, o))

                if reward is not None:
                    nextJs.append(((r, o), Q[(state, p, r, o)]))

        il += time.time() - tmp1

        tmp1 = time.time()

        Js = numpy.array([Jp for uopt, Jp in nextJs])

        temperature = T * numpy.power(0.9995, t) * 1.0 / (1 + numpy.exp(-i + 2)) + 0.10

        #print temperature, i

        exps = numpy.exp(-Js / temperature)

        probs = exps

        r = numpy.random.random() * probs.sum()

        uopt, Jp = nextJs[bisect.bisect_left(numpy.cumsum(probs), r)]

        r, o = uopt

        if t > 50:
            (r, o), Jp = sorted(nextJs, key = lambda x : x[1])[0]

        nextState, reward = getNextRealState((state, p, r, o))

        np = bisect.bisect_left(numpy.cumsum(odist), numpy.random.random())

        nextJs2 = []
        for r1 in range(4):
            for o1 in range(w):
                if (nextState, np, r1, o1) in Q:
                    nextJs2.append(((r1, o1), Q[(nextState, np, r1, o1)]))

        if len(nextJs2) > 0:
            uopt2, Jn = sorted(nextJs2, key = lambda x : x[1])[0]
        else:
            Jn = 0

        Q[(state, p, r, o)] = (1 - alpha) * Q[(state, p, r, o)] + alpha * (reward + Jn)

        #if reward < 0:
        #    1/0

        if nextState is None:
            break

        state = nextState
        p = np
        sel += time.time() - tmp1
    #1/0
    errors.append(reward + Jn)
    run_lengths.append(i)
    print t

    if t % 50 == 0:
        print t, numpy.mean(run_lengths[-50:]), numpy.mean(errors[-50:]), temperature

    #1/0
        #print rv

    #Js[i][(state, p)] = Jp
    #us[i][(state, p)] = uopt
print "Total: ", time.time() - tmp
print "Features: ", ft
print "Inner loop: ", il
print "Selection time: ", sel
#%%
for (state, p, r, o), v in Q.items():
    print (state, p, r, o), v
#%%
            for (state1, p1, r2, o2), v in Q.items():
                nextJs2 = []
                for r1 in range(4):
                    for o1 in range(w):
                        nextState1, reward = getNextRealState((state1, p1, r1, o1))

                        for np2 in range(len(pieces)):
                            if (nextState1, np2, r1, o1) in Q:
                                nextJs2.append(((r1, o1), Q[(nextState, np, r1, o1)]))

            Q[(state, p, r, o)] = (1 - alpha) * Q[(state, p, r, o)] + alpha * (reward + Jn)

        if len(nextJs2) > 0:
            uopt2, Jn = sorted(nextJs2, key = lambda x : x[1])[0]
        else:
            Jn = 0
            #print state, p, r, o
            #print 'hi'
            #1/0

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

    print 'HI', p
    for i in range(10):
        f = numpy.array(features((state, p)))

        Qc = rv.dot(f)

        nextJs = []
        for r in range(4):
            for o in range(w):
                nextState, reward = getNextRealState((state, p, r, o))

                if reward is not None:
                    nextJs.append(((r, o), Q[(state, p, r, o)]))

        (r, o), Jp = sorted(nextJs, key = lambda x : x[1])[0]

        nextState, reward = getNextRealState((state, p, r, o))

        if reward != None:
            total += reward
            if nextState != None:

                state = nextState

                r = numpy.random.random() * sum(odist)

                p = bisect.bisect_left(numpy.cumsum(odist), r)

                #print p
            else:
                #print 'Finish'
                break

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