#cython: boundscheck=False, nonecheck=False, cdivision=True
import numpy
import itertools
import bisect
import time

cdef int h = 3
cdef int w = 3

pieces = [numpy.array([[0, 1], [1, 1]]).astype('uint8'),
          numpy.array([[0, 1, 1], [1, 1, 0]]).astype('uint8'),
          numpy.array([[1], [1]]).astype('uint8')]

pieces2 = []

for piece in pieces:
    rotations = []

    rotations.append(piece)
    p = numpy.fliplr(piece).transpose()
    rotations.append(p)
    p = numpy.fliplr(p).transpose()
    rotations.append(p)
    p = numpy.fliplr(p).transpose()
    rotations.append(p)

    pieces2.append(rotations)

pieces = pieces2

gns = 0.0

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

@memoize
def getNextRealState(state, p, r, o):
    global gns
    tmp = time.time()
    p = pieces[p][r]

    estate = numpy.zeros((2 * h, w)).astype('uint8')
    estate[: h] = numpy.array(state).reshape(h, w)

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
                gns += time.time() - tmp
                return None, reward

        gns += time.time() - tmp
        return tuple(newState[:h].flatten()), reward
    else:
        gns += time.time() - tmp
        return None, None

def run(N, NT, T, lT, lI):
    #h = 14
    #w = 8

    #pieces = [numpy.array([[1, 1], [1, 1]]).astype('uint8'),
    #          numpy.array([[0, 1, 1], [1, 1, 0]]).astype('uint8'),
    #          numpy.array([[1, 1, 0], [0, 1, 1]]).astype('uint8'),
    #         numpy.array([[1], [1], [1], [1]]).astype('uint8'),
    #         numpy.array([[0, 1, 0], [1, 1, 1]]).astype('uint8'),
    #         numpy.array([[0, 0, 1], [1, 1, 1]]).astype('uint8'),
    #         numpy.array([[1, 0, 0], [1, 1, 1]]).astype('uint8')]

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

            options = sorted(nextJs.items(), key = lambda x : x[1])

            Js = numpy.array([Jp for uopt, Jp in options])

            probs = numpy.exp(-Js / T) / sum(numpy.exp(-Js / T))

            r = numpy.random.random() * probs.sum()

            uopt, Jp = options[bisect.bisect_left(numpy.cumsum(probs), r)]

            r, o = uopt

            nextState, reward = getNextRealState(state, p, r, o)

            r = numpy.random.random() * odist.sum()

            #print odist, r
            np = bisect.bisect_left(numpy.cumsum(odist), r)

            if nextState is not None:
                Qcf = rv.dot(features(nextState, np))
            else:
                Qcf = 0.0

            dt = reward + Qcf - Qc
            #print reward + Qcf, - Qc

            dr = (lT)**t * (lI)**i * dt * f / (numpy.linalg.norm(f)**2)
            rv = rv + dr

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
