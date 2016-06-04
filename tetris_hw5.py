#%%

import numpy
import itertools

#%%

h = 3
w = 3

pieces = [numpy.array([[0, 1], [1, 1]]).astype('uint8'),
          numpy.array([[0, 1, 1], [1, 1, 0]]).astype('uint8'),
          numpy.array([[1], [1]]).astype('uint8')]
#%%

def getNextState(state, p, r, o):
    piece = pieces[p]

    estate = numpy.zeros((2 * h, w)).astype('uint8')
    estate[: h] = numpy.array(state).reshape(3, 3)

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
            if newState[k, :].sum() == 3:
                removeRows.append(k)
                reward += -1

        removeRows = sorted(removeRows, reverse = True)

        for row in removeRows:
            for row2 in range(row + 1, newState.shape[0]):
                newState[row2 - 1] = newState[row2]
                newState[row2] = 0

        for i in range(h, newState.shape[0]):
            for j in range(w):
                if newState[i, j] != 0:
                    return None, reward

        return tuple(newState[:h].flatten()), reward
    else:
        return None, None


#numpy.reshape(getNextState((0, 0, 0, 0, 0, 0, 0, 0, 0), 0, 3, 1)[0], (w, h))
#%%
#%%
reachableStates = set()
for key in itertools.product([0, 1], repeat = w * h):
    state = numpy.zeros((2 * h, w)).astype('uint8')
    state[0 : h] = numpy.reshape(numpy.array(key), (h, w)).astype('uint8')

    skipState = False
    for r in range(h):
        if sum(state[r, :]) == w:
            skipState = True

    if skipState:
        continue

    reachableStates.add(key)

print len(reachableStates)
#%%
#states = set([key for source in stateSources.values()])
#%%
lut = {}
for state in reachableStates:
    for p in range(3):
        nextJs = {}

        for r in range(4):
            for o in range(w):
                nextState, reward = getNextState(state, p, r, o)

                lut[(state, p, r, o)] = nextState, reward
#%%
#N = 100
Ns = [100]#range(1, 200)
Vs = []

P = numpy.ones((h, w)) / float(w)
P = numpy.array([[0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0],
                 [1.0, 0.0, 0.0]])

for N in Ns:
    Js = {}
    us = {}
    Js[N] = {}
    us[N] = {}
    for state in reachableStates:
        for p in range(3):
            Js[N][(state, p)] = 0.0
            us[N][(state, p)] = 0

    for i in range(N - 1, -1, -1):
        Js[i] = {}
        us[i] = {}
        print i
        for state in reachableStates:
            for p in range(3):
                #nextJs = {}

                qdist = numpy.zeros((1, 3))
                qdist[0, p] = 1.0

                #P = numpy.ones((h, w)) / float(w)

                odist = qdist.dot(P)

                # Worst case analysis version:
                # We keep a nextJs seperate for each piece and then choose the one with the worst minimum to give to the user
                nextJs = { 0 : {}, 1 : {}, 2 : {} }

                for r in range(4):
                    for o in range(w):
                        nextState, reward = lut[(state, p, r, o)]#getNextState

                        if reward != None:
                            c2g = reward

                            # Worst case analysis version:
                            nextJs[0][(r, o)] = reward
                            nextJs[1][(r, o)] = reward
                            nextJs[2][(r, o)] = reward
                            if nextState != None:
                                for k in range(3):
                                    c2g += odist[0, k] * Js[i + 1][(nextState, k)]

                                # Worst case analysis version:
                                nextJs[0][(r, o)] += Js[i + 1][(nextState, 0)]
                                nextJs[1][(r, o)] += Js[i + 1][(nextState, 1)]
                                nextJs[2][(r, o)] += Js[i + 1][(nextState, 2)]

                            #nextJs[(r, o)] = c2g

                #uopt, Jp = sorted(nextJs.items(), key = lambda x : x[1])[0]

                # Worse case analysis version:
                uopt, Jp = sorted([sorted(nextJs[0].items(), key = lambda x : x[1])[0],
                                   sorted(nextJs[1].items(), key = lambda x : x[1])[0],
                                   sorted(nextJs[2].items(), key = lambda x : x[1])[0]], key = lambda x : x[1])[-1]

                Js[i][(state, p)] = Jp
                us[i][(state, p)] = uopt

    v1 = Js[0][((0, 0, 0, 0, 0, 0, 0, 0, 0), 0)]
    v2 = Js[0][((0, 0, 0, 0, 0, 0, 0, 0, 0), 1)]
    v3 = Js[0][((0, 0, 0, 0, 0, 0, 0, 0, 0), 2)]

    Vs.append((v1, v2, v3))
    print N, (v1, v2, v3)
Vs = numpy.array(Vs)
#%%
Vs = []
for i in range(200):
    v1 = Js[200 - i - 1][((0, 0, 0, 0, 0, 0, 0, 0, 0), 0)]
    v2 = Js[200 - i - 1][((0, 0, 0, 0, 0, 0, 0, 0, 0), 1)]
    v3 = Js[200 - i - 1][((0, 0, 0, 0, 0, 0, 0, 0, 0), 2)]

    Vs.append((v1, v2, v3))

Vs = numpy.array(Vs)

plt.plot(Vs[:, 0], 'r-')
plt.plot(Vs[:, 1], 'g--')
plt.plot(Vs[:, 2], 'b.')
plt.legend(('Piece 0', 'Piece 1', 'Piece 2'))
plt.xlabel('# of steps')
plt.ylabel('Number of eliminated rows')
plt.gcf().set_size_inches((10, 8))
plt.show()
#%%
state = (0, 0, 0, 0, 0, 0, 0, 0, 0)
p = 2

nextJs = {}

qdist = numpy.zeros((1, 3))
qdist[0, p] = 1.0

odist = qdist.dot(P)

for r in range(4):
    for o in range(w):
        nextState, reward = lut[(state, p, r, o)]

        if nextState == None:
            continue

        c2g = reward
        for k in range(3):
            c2g += odist[0, k] * Js[i + 1][(nextState, k)]
        nextJs[(r, o)] = c2g

uopt, Jp = sorted(nextJs.items(), key = lambda x : x[1])[0]
#%%
print getNextState(state, p, uopt[0], uopt[1])
print uopt, Jp
#%%
import bisect

vs = []

for rc in range(100):
    state, p = ((0, 0, 0, 0, 0, 0, 0, 0, 0), 0)
    total = 0.0

    for i in range(100):
        r, o = us[i][(state, p)]
        nextState, reward = lut[(state, p, r, o)]

        if nextState != None:
            state = nextState
            total += reward

            qdist = numpy.zeros((1, 3))
            qdist[0, p] = 1.0

            odist = qdist.dot(P)

            r = numpy.random.random() * sum(odist)

            p = bisect.bisect_left(numpy.cumsum(odist), r)


    vs.append(total)

print numpy.mean(vs), numpy.std(vs)
#%%
plt.hist(vs)
plt.title('Distribution of 1000 scores for 100 move game starting with piece 2')
plt.xlabel('Score')
plt.ylabel('Count')
plt.gcf().set_size_inches((6, 4))
plt.show()
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