#%%

import numpy
import itertools

#%%

h = 4
w = 4

pieces = [numpy.array([[0, 1], [1, 1]]).astype('uint8'),
          numpy.array([[0, 1, 1], [1, 1, 0]]).astype('uint8'),
          numpy.array([[1], [1]]).astype('uint8')]
#%%

def state2board(state):
    board = numpy.zeros((w, h)).astype('uint8')

    for j in range(w):
        for i in range(state[j]):
            board[i, j] = 1

    return board

def board2state(board):
    state = numpy.zeros(w).astype('uint8')

    for j in range(w):
        for i in range(h - 1, -1, -1):
            if board[i, j] != 0:
                state[j] = i + 1
                break

    return tuple(state)

print board2state(numpy.array([[1, 0, 0, 1], [0, 1, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1]]))

#%%

def getNextState(state, p, r, o):
    piece = pieces[p]

    estate = numpy.zeros((2 * h, w)).astype('uint8')
    estate[: h] = state2board(state)#numpy.array(state).reshape(3, 3)
    km = 0
    for k in range(h):
        if estate[k, :].sum() == w:
            km += 1
        else:
            break

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

        for k in range(km, estate.shape[0]):
            if newState[k, :].sum() == w:
                removeRows.append(k)
                reward += -1

        removeRows = sorted(removeRows, reverse = True)

        for row in removeRows:
            for row2 in range(row + 1, newState.shape[0]):
                newState[row2 - 1] = newState[row2]
                newState[row2] = 0

        return tuple(board2state(newState[:h])), reward
    else:
        return None, None

def getNextRealState(state, p, r, o):
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
                return None, None

        return tuple(newState[:h].flatten()), reward
    else:
        return None, None

print numpy.array(state).reshape((h, w))
print numpy.array(getNextRealState(state, p, rr, o)[0]).reshape((h, w))

#getNextState((1, 2, 3, 3), 0, 3, 1)
#%%
findColumn(state[0], pieces[p])
#%%
reachableStates = set()
for key in itertools.product([0, 1], repeat = w * h):
    #key1 = board2state(numpy.array(key).reshape((3, 3)).astype('uint8'))
    board = numpy.zeros((h, w)).astype('uint8')#state2board(key)
    board[0 : h] = numpy.reshape(numpy.array(key), (h, w)).astype('uint8')

    state = board2state(board)

    km = 0
    for k in range(h):
        if board[k, :].sum() == w:
            km += 1
        else:
            break

    skipState = False
    for r in range(km, h):
        if sum(board[r, :]) == w:
            skipState = True

    if skipState:
        continue

    reachableStates.add(state)

print len(reachableStates)
key#%%
#states = set([key for source in stateSources.values()])
#%%
lut = {}
for state in reachableStates:
    for p in range(len(pieces)):
        nextJs = {}

        for r in range(4):
            for o in range(w):
                nextState, reward = getNextState(state, p, r, o)

                lut[(state, p, r, o)] = nextState, reward
#%%
#N = 100
Ns = [20]
Vs = []

P = numpy.ones((len(pieces), len(pieces))) / float(len(pieces))
#P = numpy.array([[0.0, 1.0, 0.0],
#                 [0.0, 0.0, 1.0],
#                 [1.0, 0.0, 0.0]])

for N in Ns:
    Js = {}
    us = {}
    Js[N] = {}
    us[N] = {}
    for state in reachableStates:
        for p in range(len(pieces)):
            Js[N][(state, p)] = 0.0
            us[N][(state, p)] = 0

    for i in range(N - 1, -1, -1):
        Js[i] = {}
        us[i] = {}
        print i
        for state in reachableStates:
            for p in range(len(pieces)):
                nextJs = {}

                qdist = numpy.zeros((1, len(pieces)))
                qdist[0, p] = 1.0

                #P = numpy.ones((h, w)) / float(w)

                odist = qdist.dot(P)

                for r in range(4):
                    for o in range(w):
                        nextState, reward = lut[(state, p, r, o)]

                        if nextState == None:
                            continue

                        c2g = reward
                        for k in range(len(pieces)):
                            c2g += odist[0, k] * Js[i + 1][(nextState, k)]
                        nextJs[(r, o)] = c2g

                uopt, Jp = sorted(nextJs.items(), key = lambda x : x[1])[0]

                Js[i][(state, p)] = Jp
                us[i][(state, p)] = uopt

    v1 = Js[0][((0, 0, 0, 0), 0)]
    v2 = Js[0][((0, 0, 0, 0), 1)]
    v3 = Js[0][((0, 0, 0, 0), 2)]

    Vs.append((v1, v2, v3))
    print N, (v1, v2, v3)
Vs = numpy.array(Vs)
#%%
import bisect

vs = []

def features(state, p):
    if p == 0:
        return numpy.concatenate((state, [1, 0, 0]))
    elif p == 1:
        return numpy.concatenate((state, [0, 1, 0]))
    else:
        return numpy.concatenate((state, [0, 0, 1]))

rv = numpy.random.random(w * h + 3)
for t in range(1000):
    state, p = (tuple([0] * w * h), numpy.random.randint(0, len(pieces)))
    total = 0.0

    for i in range(N):
        f = numpy.array(features(state, p))

        Qc = max(0.0, rv.dot(f))

        nextJs = {}
        for r in range(4):
            for o in range(w):
                nextState, reward = getNextRealState(state, p, r, o)

                if nextState != None:
                    nextJs[(r, o)] = reward + Qc

        if len(nextJs) == 0:
            break
#            1/0
        #print state, len(nextJs)

        uopt, Jp = sorted(nextJs.items(), key = lambda x : x[1])[0]

        r, o = uopt
        #if i == 0:
        #    print uopt
        nextState, reward = getNextRealState(state, p, r, o)

        qdist = numpy.zeros((1, len(pieces)))
        qdist[0, p] = 1.0

        odist = qdist.dot(P)

        r = numpy.random.random() * sum(odist)

        np = bisect.bisect_left(numpy.cumsum(odist), r)
        Qcf = max(0.0, rv.dot(features(nextState, np)))
        dt = reward + Qcf - Qc
        print reward + Qcf - Qc

        #print state, nextState
        #1/0
        dr = (0.99)**(t) * dt * f / (numpy.linalg.norm(f)**2)
        rv = rv + dr

        #print numpy.linalg.norm(dr)

        state = nextState
        p = np


    #1/0
        #print rv

    #Js[i][(state, p)] = Jp
    #us[i][(state, p)] = uopt
#%%

            #board2state(numpy.array(nextState).reshape(w, h))


    r, o = us[i][(, p)]
    vs.append(total)

#%%
import bisect

vs = []

for r in range(1):
    state, p = (tuple([0] * w * h), 0)
    total = 0.0

    for i in range(50):
        f = numpy.array(features(state, p))

        Qc = rv.dot(f)

        nextJs = {}
        for r in range(4):
            for o in range(w):
                nextState, reward = getNextRealState(state, p, r, o)

                if nextState != None:
                    nextJs[(r, o)] = reward + Qc

        if len(nextJs) == 0:
            break

        (rr, o), Jp = sorted(nextJs.items(), key = lambda x : x[1])[0]

        nextState, reward = getNextRealState(state, p, rr, o)

        print numpy.array(nextState).reshape((h, w))

        #board2state(numpy.array(nextState).reshape(w, h))

        if nextState != None:
            state = nextState
            total += reward

            qdist = numpy.zeros((1, len(pieces)))
            qdist[0, p] = 1.0

            odist = qdist.dot(P)

            r = numpy.random.random() * sum(odist)

            p = bisect.bisect_left(numpy.cumsum(odist), r)

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