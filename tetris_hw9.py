#%%

import numpy
import itertools
import time
import sklearn.linear_model
import sklearn.svm

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
h = 8
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
import tensorflow as tf
sess = tf.InteractiveSession()
#%%
ILsize = 100

x = tf.placeholder(tf.float32, [None, w * h])
y_ = tf.placeholder(tf.float32, [None, 1])

W1 = tf.Variable(tf.truncated_normal([w * h, ILsize], stddev = 0.1))
b1 = tf.Variable(tf.truncated_normal([ILsize], stddev = 0.1))
W2 = tf.Variable(tf.truncated_normal([ILsize, 1], stddev = 0.1))
b2 = tf.Variable(tf.truncated_normal([1], stddev = 0.1))
y1 = tf.matmul(x, W1) + b1
e = tf.tanh(y1)
y = tf.matmul(e, W2) + b2

loss = tf.nn.l2_loss(y - y_)
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)#Adam


init = tf.initialize_all_variables()


#%%
import bisect

vs = []

ft = 0.0

N = 200

P = numpy.ones((len(pieces), len(pieces))) / float(len(pieces))
#rv = numpy.random.random(len(features(tuple([0] * w * h)))) * 0.0#01

tmp = time.time()
il = 0.0
sel = 0.0
errors = []
run_lengths = []
rewards = []
T = 100.0

#sg = sklearn.linear_model.SGDRegressor()
sg = sklearn.svm.LinearSVR()

#sg.partial_fit(numpy.array(tuple([0] * w * h)).reshape(1, -1), [-0.0])
sg.fit(numpy.array(tuple([0] * w * h)).reshape(1, -1), [-0.0])

sess.run(init)


odist = numpy.ones(len(pieces)) / len(pieces)

Qcfs = []
Qcs = []
fvs = []
for t in range(4001):
    state, p = (tuple([0] * w * h), numpy.random.randint(0, len(pieces)))#
    score = 0.0

    if len(fvs) > 1000:
        fvs = numpy.array(fvs).astype('float')
        Qcfs = numpy.array(Qcfs).reshape(-1, 1)

        for i in range(1):
          _, l = sess.run([train_step, loss], feed_dict = {x : fvs, y_ : Qcfs})

          print l
        #sg.partial_fit(numpy.array(fvs), numpy.array(Qcfs))
        #sg.fit(numpy.array(fvs), numpy.array(Qcfs))
        errors.extend(numpy.array(Qcfs) - numpy.array(Qcs))

        #if t > 500:
        #    1/0
        Qcfs = []
        Qcs = []
        fvs = []
        #print 'train'

    for i in range(N):
        f = numpy.array(state)

        #Qc = rv.dot(f)
        Qc = y.eval(feed_dict = { x : f.reshape(1, -1).astype('float')})[0][0]

        #Qc = sg.predict(f.reshape(1, -1))[0]

        tmp1 = time.time()
        nextJs = []
        for r in range(4):
            for o in range(w):
                nextState, reward = getNextRealState((state, p, r, o))

                if reward is not None:
                    total = reward

                    if nextState is not None:
                        #total += sg.predict(numpy.array(nextState).reshape(1, -1))[0]
                        total += y.eval(feed_dict = { x : numpy.array(nextState).reshape(1, -1).astype('float')})[0][0]
                        #total += rv.dot(features(nextState))

                    nextJs.append(((r, o), total))

        il += time.time() - tmp1

        tmp1 = time.time()

        #if t > 50:
        #    (r, o), Jp = sorted(nextJs, key = lambda x : x[1])[0]
        #else:
        if True:
            Js = numpy.array([Jp for uopt, Jp in nextJs])
            temperature = T * numpy.power(0.99, t) * 1.0 / (1 + numpy.exp(-i + 2)) + 0.1
            exps = numpy.exp(-Js / temperature)
            probs = exps
            r = numpy.random.random() * probs.sum()
            uopt, Jp = nextJs[bisect.bisect_left(numpy.cumsum(probs), r)]
            r, o = uopt

        nextState, reward = getNextRealState((state, p, r, o))

        r = numpy.random.random()# * odist.sum() -- this sum is always 1.0

        np = bisect.bisect_left(numpy.cumsum(odist), r)

        if nextState is not None:
            #Qcf = sg.predict(numpy.array(nextState).reshape(1, -1))[0]
            Qcf = y.eval(feed_dict = { x : numpy.array(nextState).reshape(1, -1).astype('float')})[0][0]
        else:
            Qcf = 0.0

        score += reward

        #dt = reward + 0.9 * Qcf - Qc

        Qcfs.append(reward + 1.0 * Qcf)
        Qcs.append(Qc)
        fvs.append(f)

        if nextState is None:
            break

        state = nextState
        p = np
        sel += time.time() - tmp1

    run_lengths.append(i)
    rewards.append(score)

    if t % 50 == 0:
        print t, numpy.mean(run_lengths[-50:]), numpy.mean(errors[-50:]), numpy.mean(rewards[-50:]), temperature

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
        Qc = sg.predict(numpy.array(state).reshape(1, -1))[0]

        nextJs = []
        for r in range(4):
            for o in range(w):
                nextState, reward = getNextRealState((state, p, r, o))

                if reward is not None:
                    t = reward

                    if nextState is not None:
                        t += sg.predict(numpy.array(nextState).reshape(1, -1))[0]

                    nextJs.append(((r, o), t))

        (r, o), Jp = sorted(nextJs, key = lambda x : x[1])[0]

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