#%%

import numpy
import itertools
import time


#%%

h = 3
w = 3

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
h = 20
w = 8

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
    pls = []

    for j in range(p.shape[1]):
        for i in range(p.shape[0]):
            if p[i, j] == 1:
                break
        pls.append(-i)

    hs = []

    for j in range(w):
        i = h - 1
        while i >= -1:
            if i == -1:
                hs.append(0)
                break

            if estate[i, j] > 0:
                hs.append(i + 1)
                break
            i -= 1

    if o <= w - p.shape[1]:
        of = []
        for j in range(p.shape[1]):
            of.append(hs[o + j] + pls[j])

        d = max(of)

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
nextState, reward = getNextRealState(((0, 1, 1, 0, 0, 0, 0, 0, 0), 0, 0, 0))

print numpy.reshape(nextState, (h, w))
#%%
import bisect

vs = []

ft = 0.0
@memoize2
def features(args):
    state, p = args
    heights = numpy.zeros((h, w))

    state = numpy.array(state).reshape(h, w)

    hs = []

    for j in range(w):
        i = h - 1
        while i >= -1:
            if i == -1:
                hs.append(0)
                break
            if state[i, j] == 1:
                hs.append(i + 1)
                break
            i -= 1

    dhs = []
    for i in range(len(hs) - 1):
        dhs.append(hs[i + 1] - hs[i])

    features = [1.0]

    features.extend(hs)
    #for hi in hs:
    #    v = numpy.zeros(h + 1)
    #    v[hi] = 1.0
    #    features.extend(v)

    for hi in dhs:
    #    v = numpy.zeros(h + 1)
    #    v[numpy.abs(hi)] = 1.0
        features.append(numpy.abs(hi))

    features.append(max(hs))

    holes = 0.0
    for j in range(w):
        hi = hs[j]
        for i in range(hi):
            if state[i, j] == 0.0:
                holes += 1

    features.append(holes)
    #state, p = args
    #state = args
    #morestate = []
    #morestate2 = []
    #morestate3 = []
    #morestate4 = []

    #for i in range(len(state) - 1):
    #    morestate.append(state[i] | state[i + 1])
    #    morestate.append(state[i] ^ state[i + 1])
    #    morestate.append(state[i] & state[i + 1])

    #state = numpy.array(state)

    #state = numpy.concatenate((state, morestate))#

    #out = state
    #piece = numpy.zeros(len(pieces))
    #piece[p] = 1.0

    #out = numpy.concatenate((features, piece))# - 0.5

    #features.append(1.0)

    return numpy.array(features)
#print features((state, p))
#print numpy.reshape(state, (h, w))

N = 200

P = numpy.ones((len(pieces), len(pieces))) / float(len(pieces))
rv = numpy.random.random(len(features((tuple([0] * w * h), 0)))) * 0.1

rv[: w] = 0
rv[w : 2 * w] = 0
rv[-1] = 1
rv[-2] = 10

#rv[:30] = -1
#rv[30:74] = -1
#rv[:10] = -10

tmp = time.time()
il = 0.0
sel = 0.0
trn = 0.0
errors = []
run_lengths = []
rewards = []
T = 0.25
odist = numpy.ones(len(pieces)) / len(pieces)

import sklearn.linear_model
lr = sklearn.linear_model.LinearRegression(fit_intercept = False)#alpha = 1.0,
rb = 0.0
#alpha = 1.0,
guess_next = 0.0
recordings = []
for t in range(501):
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

    state, p = (tuple([0] * w * h), numpy.random.randint(0, len(pieces)))#
    score = 0.0

    for i in range(N):
        f = numpy.array(features((state, p)))
        Qc = rv.dot(f)

        nextJs = []
        tmp1 = time.time()
        for r in range(4):
            for o in range(w):
                nextState, reward = getNextRealState((state, p, r, o))

                if reward is not None:
                    total = reward

                    if nextState is not None:
                        #for np in range(len(pieces)):
                        total += (rv.dot(features((nextState, np))))

                        nextJs.append(((r, o), total))
        il += time.time() - tmp1

        if len(nextJs) == 0:
            break

        (r, o), Jp = sorted(nextJs, key = lambda x : x[1])[0]

        nextState, reward = getNextRealState((state, p, r, o))

        #print (r, o), nextJs
        #print numpy.reshape(nextState, (h, w))
        #print pieces[p]
        #print numpy.reshape(state, (h, w))
        #print '----'

        r = numpy.random.random()
        np = bisect.bisect_left(numpy.cumsum(odist), r)

        score += reward

        if nextState is not None:
            Qcf = rv.dot(features((nextState, np)))
        else:
            Qcf = 0.0
            break

        dt = reward + Qcf - Qc

        recording.append((f, Qc, dt))

        #if nextState is None:
        #    break

        state = nextState
        p = np

    run_lengths.append(i)
    rewards.append(score)
    recordings.append(recording)

    if t % 1 == 0:
        print t, numpy.mean(run_lengths), numpy.mean(rewards)

    #1/0
        #print rv

    #Js[i][(state, p)] = Jp
    #us[i][(state, p)] = uopt

print "Total: ", time.time() - tmp
print "Features: ", ft
print "Inner loop: ", il
print "Selection time: ", sel
print "Train time: ", trn
#%%
p = 0
#[[1 1]
# [1 1]]
state = (0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

nextJs = []
for r in range(4):
    for o in range(w):
        nextState, reward = getNextRealState((state, p, r, o))

        if nextState:
            print numpy.reshape(nextState, (h, w))

#%%

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

        #print r, o, nextJs
        #1/0

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