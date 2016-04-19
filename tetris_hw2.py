#%%

import numpy

#%%

h = 3
w = 3

state = numpy.zeros((2 * h, w))

pieces = [numpy.array([[0, 1], [1, 1]]).astype('uint8'), numpy.array([[0, 1, 1], [1, 1, 0]]).astype('uint8'), numpy.array([[1], [1]]).astype('uint8')]
#%%

def findColumn(state, piece):
    estate = numpy.zeros((2 * h, w)).astype('uint8')
    estate[: h] = numpy.array(state).reshape(3, 3)

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
            #print o, d, p.shape, newState.shape
            newState[d : d + p.shape[0], o : o + p.shape[1]] += p

            reward = 0
            removeRows = []

            for k in range(0, estate.shape[0]):
                if newState[k, :].sum() == 3:
                    removeRows.append(k)
                    #print 'hi'
                    reward += -1

            #print newState

            removeRows = sorted(removeRows, reverse = True)

            #print removeRows

            for row in removeRows:
                for row2 in range(row + 1, newState.shape[0]):
                    newState[row2 - 1] = newState[row2]
                    newState[row2] = 0

            #print newState
            #print '*'

            #print '----'

            options.append((newState, (r, o), reward))

    return options
#%%
findColumn(state, p)
#%%
findColumn(state[0], pieces[p])
#%%
def reverseColumn(state, piece):
    estate = numpy.ones((3 * h, w))

    estate[h : 3 * h, :w] = state

    p0 = piece
    p1 = numpy.fliplr(p0).transpose()
    p2 = numpy.fliplr(p1).transpose()
    p3 = numpy.fliplr(p2).transpose()

    options = []

    for (r, p) in [(0, p0), (1, p1), (2, p2), (3, p3)]:
        for o in range(0, 1 + w - p.shape[1]):
            for d in range(3 * h - p.shape[0], -1, -1):
                #print p.shape, estate.shape, d
                overlap = (estate[d : d + p.shape[0], o : o + p.shape[1]] * p).flatten().sum()
                if overlap == p.flatten().sum():
                    #print '*****'
                    #print estate
                    #print p.flatten().sum()
                    #print overlap
                    #print o, d, p
                    newState = numpy.array(estate).astype('uint8')

                    newState[d : d + p.shape[0], o : o + p.shape[1]] -= p
                    #print newState
                    if not newState[d - 1].any():
                        #print 'hi'
                        continue

                    if newState[2 * h : 3 * h].flatten().sum() != 0:
                        #print 'hi2'
                        continue

                    upShift = max(0, h - d)

                    #print upShift

                    if upShift > 0:
                        newState[upShift :] = newState[: -upShift]

                    #print newState, d, p.shape[0], 1, 2 * h
                    #print '-----'

                    if newState[2 * h : 3 * h].flatten().sum() != 0:
                        #print 'hi3'
                        continue

                    #1/0
                    #print '--'
                    #print estate
                    #print newState
                    #print p, 'r', r, 'u', upShift, d, p.shape
                    #print estate[d : d + p.shape[0]], d
                    #print '**'
                    options.append((newState[h : 2 * h], (r, o), -upShift))

                    break

                #elif overlap > 0:
                #    break

    return options
#%%
reverseColumn(state, p)
#%%
stateSources = {}
for i in range(2**9):
    key = (i & 2**0, (i & 1 << 1) >> 1, (i & 1 << 2) >> 2, (i & 1 << 3) >> 3, (i & 1 << 4) >> 4, (i & 1 << 5) >> 5, (i & 1 << 6) >> 6, (i & 1 << 7) >> 7, (i & 1 << 8) >> 8)
    state = numpy.zeros((2 * h, w)).astype('uint8')
    state[0 : h] = numpy.reshape(numpy.array(key), (h, w)).astype('uint8')

    for p in pieces:
        options = reverseColumn(state, p)
        for newState, (r, o), reward in options:
            if key not in stateSources:
                stateSources[key] = []

            nkey = tuple(newState[0 : h].flatten())
            #1/0

            stateSources[key].append((nkey, state, newState, p, (r, o), reward))
            #print key, nkey
            #print state
            #print newState
            #print p
            #print (r, o)
            #print reward
            #1/0
       # 1/0
    #endStates[nkey] = (nState, p, action, reward)
#%%
reachableStates = set()
for i in range(2**9):
    key = (i & 2**0, (i & 1 << 1) >> 1, (i & 1 << 2) >> 2, (i & 1 << 3) >> 3, (i & 1 << 4) >> 4, (i & 1 << 5) >> 5, (i & 1 << 6) >> 6, (i & 1 << 7) >> 7, (i & 1 << 8) >> 8)
    state = numpy.zeros((2 * h, w)).astype('uint8')
    state[0 : h] = numpy.reshape(numpy.array(key), (h, w)).astype('uint8')

    skipState = False
    for r in range(h):
        if sum(state[r, :]) == w:
            skipState = True

    if skipState:
        continue

    for p in pieces:
        options = findColumn(key, p)
        for newState, (r, o), reward in options:
            nkey = tuple(newState[0 : h].flatten())
            reachableStates.add(nkey)
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
for state, p in ind2state:
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
            stateOptions[(state, p)][(r, o)] = (0, 0)
#%%
sources = {}
for (state, p), options in stateOptions.items():
    for action, (nkey, reward) in options.items():
        if nkey not in sources:
            sources[nkey] = set()

        sources[nkey].add(((state, p), action, reward))

for nkey in sources:
    sources[nkey] = list(sources[nkey])

#states = set([key for source in stateSources.values()])
#%%
N = 100
stages = {}
stages[N - 1] = {}
for source in sources.keys() + [0]:#
    for (state, p), action, reward in sources[source]:
        if True:#p == (N - 1) % 3:
            if (state, p) not in stages[N - 1]:
                stages[N - 1][(state, p)] = []

            stages[N - 1][(state, p)].append((reward, action, 0))

for stage in range(N - 2, -1, -1):
    stages[stage] = {}

    for target in stages[stage + 1]:
        if target[0] not in sources:
            continue

        for (state, p), action, reward in sources[target[0]]:
            if True:#p == stage % 3 or stage == 0:
                if (state, p) not in stages[stage]:
                    stages[stage][(state, p)] = []

                stages[stage][(state, p)].append((reward, action, target[0]))

    if True:
        target = (0, 0)
        for (state, p), action, reward in sources[target[0]]:
            if True:#p == stage % 3 or stage == 0:
                if (state, p) not in stages[stage]:
                    stages[stage][(state, p)] = []

                stages[stage][(state, p)].append((reward, action, target[0]))
    #print 'hi'

for stage in range(N):
    print len(stages[stage]), stage
#%%
for state, p  in stages[N - 2]:
    print stages[N - 2][(state, p)]
    print '---'
#%%
paths = {}
for stage in range(N - 1, -1, -1):
    paths[stage] = {}
    #print '----'
    for (state, p) in stages[stage]:
        #options = filter(lambda x : x[2][1] == np, options)

        #opponent = paths[stage + 1][(options[0][2], np)]

        options = sorted(stages[stage][(state, p)], key = lambda x : x[0])

        if True:
            #np = (stage + 1) % 3

            rewards = {}
            opts = {}
            for option in options:
                for np in range(len(pieces)):
                    if np not in rewards:
                        rewards[np] = []
                        opts[np] = []

                    if stage + 1 in paths:
                        c2g = paths[stage + 1][(option[2], np)][0]
                    else:
                        c2g = 0

                    #if state == (0, 0, 0, 0, 0, 0, 0, 0, 0) and np == 1 and p == 1:
                        #print options[0], c2g
                        #if stage + 1 in paths:
                        #    print paths[stage + 1][(option[2], np)][0]
                        #    print option[2], np
                        #    print '**'

                    rewards[np].append(options[0][0] + c2g)#
                    opts[np].append(option)

            #if state == (0, 0, 0, 0, 0, 0, 0, 0, 0):
            #    print rewards

            for np in rewards:
                idx = numpy.argmin(rewards[np])
                rewards[np] = rewards[np][idx]
                opts[np] = opts[np][idx]

            np = sorted(rewards.items(), key = lambda x : x[1], reverse = True)[0][0]

            #np = (stage + 1) % 3

            if stage + 1 in paths:
                c2g = paths[stage + 1][(opts[np][2], np)][0]
            else:
                c2g = 0

            #if state == (0, 0, 0, 0, 0, 0, 0, 0, 0):
            #    if stage + 1 in paths:
            #        print paths[stage + 1][(options[0][2], 0)]
            #        print paths[stage + 1][(options[0][2], 1)]
            #        print paths[stage + 1][(options[0][2], 2)]
            #    print 'stage ', stage, ' rewards ', rewards, ' p ', p
            #    print 'np', np
            #    print 'state' , state, 'p', p
            #    print 'c2g', c2g, options[0][0]


            #np = (stage + 1) % 3

            #if rewards[np] >= paths[stage + 1][(options[0][2], np)][0]:
            #    print 'hi'
            #1/0

            #if stage + 1 in paths:
            #    filteredOptions = []
            #    for option in options:
            #        if (option[2], np) in paths[stage + 1]:
            #            filteredOptions.append(option)#nps.add(np)

            #    options = sorted(filteredOptions, key = lambda x : x[0])
            #if (options[0][2], np) in paths[stage + 1]:
            #else:
            #    print 'hihihihihi'
            #    c2g = 0

            paths[stage][(state, p)] = (opts[np][0] + c2g, (opts[np][2], np))
        else:
            c2g = 0

            paths[stage][(state, p)] = (options[0][0], (options[0][2], np))
#%%
print 'e', paths[0][((0, 0, 0, 0, 0, 0, 0, 0, 0), 0)]
print 'f1', paths[0][((0, 0, 0, 0, 0, 0, 0, 0, 0), 0)]
print 'f2', paths[0][((0, 0, 0, 0, 0, 0, 0, 0, 0), 1)]
print 'f3', paths[0][((0, 0, 0, 0, 0, 0, 0, 0, 0), 2)]
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
states = {}
for rkey, (state, action, reward) in endStates.items():
    #print state
    for p in pieces:
        options = reverseColumn(state, p)

        for nState, action2, reward2 in options:
            nkey = tuple(nState[0 : h].flatten())

            if nkey in states:
                if states[nkey][2] < reward2:
                    print 'whats up'
                    continue

            states[nkey] = (nState, p, action2, reward2)

        #print p, options
#%%

1/0
#%%
endStates = {}
for i in range(2**9):
    key = (i & 2**0, (i & 1 << 1) >> 1, (i & 1 << 2) >> 2, (i & 1 << 3) >> 3, (i & 1 << 4) >> 4, (i & 1 << 5) >> 5, (i & 1 << 6) >> 6, (i & 1 << 7) >> 7, (i & 1 << 8) >> 8)
    state = numpy.zeros((2 * h, w)).astype('uint8')
    state[0 : h] = numpy.reshape(numpy.array(key), (h, w)).astype('uint8')

    for p in pieces:
        options = findColumn(state, p)

        for nState, action, reward in options:
            if nState[h : 2 * h].flatten().sum() > 0:
                nkey = tuple(nState[0 : h].flatten())

                if nkey in endStates:
                    if endStates[nkey][3] < reward:
                        #print 'whats up', endStates[nkey][3], reward

                        #if endStates[nkey][3] == -2:
                        #    1/0
                        continue

                #if reward == -2:
                #    1/0

                endStates[nkey] = (nState, p, action, reward)
#%%

for i in range(0, 2):
    piece = i % 3

    out = reverseColumn(state, pieces[piece])#findColumn