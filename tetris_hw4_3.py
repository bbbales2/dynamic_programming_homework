#%%

import numpy
import itertools

#%%

h = 5
w = 5

state = numpy.zeros((2 * h, w))

pieces = [numpy.array([[1, 1], [1, 1]]).astype('uint8'),
          numpy.array([[0, 1, 1], [1, 1, 0]]).astype('uint8'),
          numpy.array([[1, 1, 0], [0, 1, 1]]).astype('uint8'),
         numpy.array([[1], [1], [1], [1]]).astype('uint8'),
         numpy.array([[0, 1, 0], [1, 1, 1]]).astype('uint8'),
         numpy.array([[0, 0, 1], [1, 1, 1]]).astype('uint8'),
         numpy.array([[1, 0, 0], [1, 1, 1]]).astype('uint8')]
#%%

def findColumn(state, piece):
    estate = numpy.zeros((2 * h, w)).astype('uint8')
    estate[: h] = numpy.array(state).reshape(h, w)

    p0 = piece
    p1 = numpy.fliplr(p0).transpose()
    p2 = numpy.fliplr(p1).transpose()
    p3 = numpy.fliplr(p2).transpose()

    options = []

    for r, p in [(0, p0), (1, p1), (2, p2), (3, p3)]:
        for o in range(0, 1 + w - p.shape[1]):
            for d in range(2 * h - p.shape[0], -1, -1):
                if (estate[d : d + p.shape[0], o : o + p.shape[1]] * p).flatten().sum() != 0:
                    break

                lastd = d

            d = lastd
            newState = numpy.array(estate)
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

            options.append((newState, (r, o), reward))

    return options
#%%
findColumn(state, p)
#%%
findColumn(state[0], pieces[p])
#%%
reachableStates = set()
for i in range(2**16):
    key = (i & 2**0, (i & 1 << 1) >> 1, (i & 1 << 2) >> 2, (i & 1 << 3) >> 3, (i & 1 << 4) >> 4, (i & 1 << 5) >> 5, (i & 1 << 6) >> 6, (i & 1 << 7) >> 7, (i & 1 << 8) >> 8, (i & 1 << 9) >> 9, (i & 1 << 10) >> 10, (i & 1 << 11) >> 11, (i & 1 << 12) >> 12, (i & 1 << 13) >> 13, (i & 1 << 14) >> 14, (i & 1 << 15) >> 15, (i & 1 << 16) >> 16, (i & 1 << 17) >> 17, (i & 1 << 18) >> 18, (i & 1 << 19) >> 19, (i & 1 << 20) >> 20, (i & 1 << 21) >> 21, (i & 1 << 22) >> 22, (i & 1 << 23) >> 23, (i & 1 << 24) >> 24)
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
while True:
    reachableStatesOld = set(reachableStates)

    reachableStates = set()

    for key in reachableStatesOld:
        for p in pieces:
            options = findColumn(key, p)
            for newState, (r, o), reward in options:
                nkey = tuple(newState[0 : h].flatten())
                reachableStates.add(nkey)

    print len(reachableStates)

    if reachableStates == reachableStatesOld:
        break
#%%
ind2state = []
for state in reachableStates:
    for p in range(len(pieces)):
        ind2state.append((state, p))

for p in range(len(pieces)):
    ind2state.append((0, p))

state2ind = dict([(v, k) for k, v in enumerate(ind2state)])

#%%
stateOptions = {}
for i, (state, p) in enumerate(ind2state):
    stateOptions[(state, p)] = {}

    if state == 0:
        stateOptions[(state, p)][(0, 0)] = (0, 0)

        continue

    options = findColumn(state, pieces[p])

    for nState, (r, o), reward in options:
        if nState[h : 2 * h].flatten().sum() == 0:
            nkey = tuple(nState[0 : h].flatten())

            stateOptions[(state, p)][(r, o)] = (nkey, reward)#[((nkey, p1), reward) for p1 in range(len(pieces))]
        else:
            stateOptions[(state, p)][(r, o)] = (0, reward)

    print i, len(ind2state)
#%%
sources = {}
for i, ((state, p), options) in enumerate(stateOptions.items()):
    for action, (nkey, reward) in options.items():
        if nkey not in sources:
            sources[nkey] = set()

        sources[nkey].add(((state, p), action, reward))

    print i, len(stateOptions)

for nkey in sources:
    sources[nkey] = list(sources[nkey])

#states = set([key for source in stateSources.values()])
#%%
#N = 100
Ns = [10]
Vs = []
for N in Ns:
    stages = {}
    stages[N - 1] = {}
    for target in sources.keys() + [0]:#
        for (state, p), action, reward in sources[target]:
            if True:#p == (N - 1) % 3:
                if (state, p) not in stages[N - 1]:
                    stages[N - 1][(state, p)] = set()

                stages[N - 1][(state, p)].add((reward, action, target))

    for stage in range(N - 2, -1, -1):
        stages[stage] = {}

        for target in stages[stage + 1]:#itertools.chain(, [(0, 0)]):
            if target[0] not in sources:
                continue

            for (state, p), action, reward in sources[target[0]]:
                if True:#p == stage % 3 or stage == 0:
                    if (state, p) not in stages[stage]:
                        stages[stage][(state, p)] = set()

                    stages[stage][(state, p)].add((reward, action, target[0]))
        print stage

    #for stage in range(N):
    #    print len(stages[stage]), stage
    ##%%
    #for state, p  in stages[N - 2]:
    #    print stages[N - 2][(state, p)]
    #    print '---'
    ##%%
    paths = {}
    for stage in range(N - 1, -1, -1):
        paths[stage] = {}
        for (state, p) in stages[stage]:
            options = sorted(stages[stage][(state, p)], key = lambda x : x[0])

            nstates = set([option[2] for option in options])
            nstateRewards = {}
            for option in options:
                if option[2] not in nstateRewards:
                    nstateRewards[option[2]] = []#option[0]
                nstateRewards[option[2]].append(option)

            for nstate in nstateRewards:
                nstateRewards[nstate] = min([o[0] for o in nstateRewards[nstate]])

            #print nstateRewards
            qdist = numpy.zeros((1, 7))
            qdist[0, p] = 1.0

            #P = numpy.array([[0.0, 1.0, 0.0],
            #                 [0.0, 1.0, 0.0],
            #                 [0.0, 1.0, 0.0]])

            #P = numpy.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            #                 [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            #                 [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]])

            P = numpy.ones((7, 7)) / 49.0

            odist = qdist.dot(P)

            rewards = []

            for nstate in nstates:
                c2g = 0.0
                if stage + 1 in paths:
                    for np in range(len(pieces)):
                        c2g += odist[0, np] * paths[stage + 1][(nstate, np)][0]
                rewards.append((nstate, nstateRewards[nstate] + c2g))

            rewards = sorted(rewards, key = lambda x : x[1])

            nstate, total = rewards[0]

            for np in range(len(pieces)):
                paths[stage][(state, p)] = (total, (nstate, np))
        print stage


    v1 = paths[0][((0, 0, 0, 0, 0, 0, 0, 0, 0), 0)][0]
    v2 = paths[0][((0, 0, 0, 0, 0, 0, 0, 0, 0), 1)][0]
    v3 = paths[0][((0, 0, 0, 0, 0, 0, 0, 0, 0), 2)][0]

    Vs.append((v1, v2, v3))
    print N, (v1, v2, v3)
    print min([value[0] for value in paths[0].values()])
Vs = numpy.array(Vs)
#%%
#print 'e', paths[0][((0, 0, 0, 0, 0, 0, 0, 0, 0), 0)]
#print 'N',
print 'f1', paths[0][((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), 0)]
print 'f2', paths[0][((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), 1)]
print 'f3', paths[0][((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), 2)]
print 'f3', paths[0][((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), 3)]
print 'f3', paths[0][((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), 4)]
print 'f3', paths[0][((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), 5)]
print 'f3', paths[0][((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), 6)]
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