#cython: boundscheck=False, nonecheck=False, cdivision=True
import numpy
import itertools
import bisect
import time

cimport numpy

from cython import array

h_ = 14
w_ = 8

pieces = [numpy.array([[1, 1], [1, 1]]).astype('int'),
          numpy.array([[0, 1, 1], [1, 1, 0]]).astype('int'),
          numpy.array([[1, 1, 0], [0, 1, 1]]).astype('int'),
         numpy.array([[1], [1], [1], [1]]).astype('int'),
         numpy.array([[0, 1, 0], [1, 1, 1]]).astype('int'),
         numpy.array([[0, 0, 1], [1, 1, 1]]).astype('int'),
         numpy.array([[1, 0, 0], [1, 1, 1]]).astype('int')]

#h_ = 3
#w_ = 3

#pieces = [numpy.array([[0, 1], [1, 1]]).astype('int'),
#          numpy.array([[0, 1, 1], [1, 1, 0]]).astype('int'),
#          numpy.array([[1], [1]]).astype('int')]


cdef int h = h_
cdef int w = w_

pieces2 = []

for p, piece in enumerate(pieces):
    ps = []

    p1 = piece.copy()
    ps.append(p1)

    p2 = numpy.fliplr(p1).transpose()
    ps.append(p2)

    p3 = numpy.fliplr(p2).transpose()
    ps.append(p3)

    p4 = numpy.fliplr(p3).transpose()
    ps.append(p4)

    ps2 = []
    for piece in ps:
        pls = []

        for j in range(piece.shape[1]):
            for i in range(piece.shape[0]):
                if piece[i, j] == 1:
                    break

            pls.append(-i)
        ps2.append((piece, pls))

    pieces2.append(ps2)

pieces = pieces2

def getHeight(state):
    state = numpy.array(state).reshape(h, w)

    hs = [0] * w

    for j in range(w):
        for i in range(h - 1, -1, -1):
            if state[i, j] > 0:
                hs[j] = i + 1
                break

    return hs

#@memoize2
def getNextRealState(args):
    #tmp1 = time.time()

    cdef int pi, r, d, o, i, j, k, nrowsr, row, row2, ps0, ps1
    cdef float total, reward
    cdef numpy.ndarray[numpy.int_t, ndim = 2] estate, p

    state, pi, r, o, hs = args
    p, pls = pieces[pi][r]

    ps0 = p.shape[0]
    ps1 = p.shape[1]

    if o <= w - ps1:
        estate = numpy.zeros((4 + h, w), dtype = numpy.int)
        estate[:h] = numpy.array(state).reshape(h, w)

        of = []
        for j in range(ps1):
            of.append(hs[o + j] + pls[j])

        d = max(of)

        for i in range(ps0):
            for j in range(ps1):
                estate[d + i, o + j] += p[i, j]
        #newState[d : d + p.shape[0], o : o + p.shape[1]] += p

        reward = 0
        removeRows = []
        nrowsr = 0

        for k in range(d, d + ps0):
            total = 0.0
            for j in range(w):
                total += estate[k, j]

            if total == w:
                removeRows.append(k)
                nrowsr += 1
                reward += -1

        removeRows = sorted(removeRows, reverse = True)

        for i in range(nrowsr):
            row = removeRows[i]
            for row2 in range(row + 1, h):
                estate[row2 - 1] = estate[row2]
                estate[row2] = 0

        for i in range(h, 4 + h):
            for j in range(w):
                if estate[i, j] != 0:
                    return None, reward

        out = []
        for i in range(h):
            for j in range(w):
                out.append(estate[i, j])

        #print time.time() - tmp1
        return tuple(out), reward
    else:
        #print time.time() - tmp1
        return None, None

def features(args):
    state_, p = args

    cdef int i, j, hi, holes
    cdef numpy.ndarray[numpy.int_t, ndim = 2] state
    cdef numpy.ndarray[numpy.float_t, ndim = 1] features

    features = numpy.zeros(2 * w + 2)
    state = numpy.zeros((h, w), dtype = numpy.int)

    for i in range(h):
        for j in range(w):
            state[i, j] = state_[i * w + j]

    cdef numpy.ndarray[numpy.int_t, ndim = 1] hs
    hs = numpy.zeros(w, dtype = numpy.int)

    for j in range(w):
        for i in range(h - 1, -1, -1):
            if state[i, j] > 0:
                hs[j] = i + 1
                break

    dhs = []
    features[0] = 1.0

    for i in range(w):
        features[1 + i] = hs[i]

    for i in range(w - 1):
        features[1 + w + i] = abs(hs[i + 1] - hs[i])

    features[2 * w] = max(hs)

    holes = 0
    for j in range(w):
        hi = hs[j]
        for i in range(hi):
            if state[i, j] == 0:
                holes += 1

    features[2 * w + 1] = float(holes)

    return features

def run(N, NT, T, lT, lI):
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
