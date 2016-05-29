import numpy
import itertools
import bisect

def run(N, NT, T, lT, lI):
    h = 14
    w = 8

    pieces = [numpy.array([[1, 1], [1, 1]]).astype('uint8'),
              numpy.array([[0, 1, 1], [1, 1, 0]]).astype('uint8'),
              numpy.array([[1, 1, 0], [0, 1, 1]]).astype('uint8'),
             numpy.array([[1], [1], [1], [1]]).astype('uint8'),
             numpy.array([[0, 1, 0], [1, 1, 1]]).astype('uint8'),
             numpy.array([[0, 0, 1], [1, 1, 1]]).astype('uint8'),
             numpy.array([[1, 0, 0], [1, 1, 1]]).astype('uint8')]


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
                    return None, reward

            return tuple(newState[:h].flatten()), reward
        else:
            return None, None

    import bisect

    vs = []

    def features(state, p):
        morestate = []
    #
        for i in range(len(state) - 1):
            morestate.append(state[i] * state[i + 1])

        state = numpy.array(state) - 0.5

        state = numpy.concatenate((state, morestate))

        nstate = [[0] * len(state)] * len(pieces)
        nstate[p] = state

        return numpy.concatenate(nstate)

    P = numpy.ones((len(pieces), len(pieces))) / float(len(pieces))
    rv = numpy.random.random(len(features([0] * w * h, numpy.random.randint(0, len(pieces))))) * 0.0

    errors = []

    cdef int r, o, np, t, i
    for t in range(NT):
        state, p = (tuple([0] * w * h), 0)#numpy.random.randint(0, len(pieces))
        total = 0.0

        for i in range(N):
            f = numpy.array(features(state, p))

            Qc = rv.dot(f)

            qdist = numpy.zeros((1, len(pieces)))
            qdist[0, p] = 1.0

            odist = qdist.dot(P)

            nextJs = {}
            for r in range(4):
                for o in range(w):
                    nextState, reward = getNextRealState(state, p, r, o)

                    if reward is not None:
                        nextJs[(r, o)] = reward

                        if nextState is not None:
                            for np in range(3):
                                nf = numpy.array(features(nextState, np))

                                nextJs[(r, o)] += odist[0, np] * rv.dot(nf)

                    #if nextState != None:
                    #    nextJs[(r, o)] = reward + Qc

    #            1/0
            #print state, nextJs

            options = sorted(nextJs.items(), key = lambda x : x[1])

            Js = numpy.array([Jp for uopt, Jp in options])

            probs = numpy.exp(-Js / T) / sum(numpy.exp(-Js / T))

            #print probs
            #if t == 25 and i == 5:
            #    1/0

            r = numpy.random.random() * sum(probs)

            uopt, Jp = options[bisect.bisect_left(numpy.cumsum(probs), r)]

            #if numpy.random.rand() < 0.5:
            #    uopt, Jp = options[0]
            #else:
            #    uopt, Jp = options[numpy.random.randint(len(options))]
            #1/0
            r, o = uopt
            #print t, i, p, r, o
            #if i == 0:
            #    print uopt
            nextState, reward = getNextRealState(state, p, r, o)

            r = numpy.random.random() * odist.sum()

            #print odist, r
            np = bisect.bisect_left(numpy.cumsum(odist), r)

            if nextState is not None:
                Qcf = rv.dot(features(nextState, np))
            else:
                Qcf = 0.0

            dt = reward + Qcf - Qc
            #if (state, p):
            #    print dt
            #print reward + Qcf, - Qc
            #print numpy.linalg.norm(f)**2
            #1/0
            #errors.append(dt)

            #print state, nextState
            #1/0
            dr = (lT)**t * (lI)**i * dt * f / (numpy.linalg.norm(f)**2)
            rv = rv + dr

            #print numpy.linalg.norm(dr)

            if nextState is None:
                #print i, (0.75)**i
                errors.append(i)
                break

            state = nextState
            p = np
        print rv
        #plt.plot(errors)
        #plt.show()
        print 'hi', numpy.mean(errors[-20:])

    vs = []

    for r in range(NT):
        state, p = (tuple([0] * w * h), numpy.random.randint(len(pieces)))
        total = 0.0

        #print 'HI', p
        for i in range(N):
            f = numpy.array(features(state, p))

            Qc = rv.dot(f)

            nextJs = {}
            for r in range(4):
                for o in range(w):
                    nextState, reward = getNextRealState(state, p, r, o)

                    if reward is not None:
                        nextJs[(r, o)] = reward

                        if nextState is not None:
                            for np in range(3):
                                nf = numpy.array(features(nextState, np))

                                nextJs[(r, o)] += odist[0, p] * rv.dot(nf)

            (rr, o), Jp = sorted(nextJs.items(), key = lambda x : x[1])[0]

            nextState, reward = getNextRealState(state, p, rr, o)

            #if nextState != None:
             #   print nextJs, rr, o
            #    print numpy.array(nextState).reshape((h, w))

            #board2state(numpy.array(nextState).reshape(w, h))

            if reward != None:
                total += reward
                if nextState != None:

                    state = nextState

                    qdist = numpy.zeros((1, len(pieces)))
                    qdist[0, p] = 1.0

                    odist = qdist.dot(P)

                    r = numpy.random.random() * odist.sum()

                    p = bisect.bisect_left(numpy.cumsum(odist), r)

                    #print p
                else:
                    #print 'Finish'
                    break

        vs.append(total)

    print numpy.mean(vs), numpy.std(vs)
