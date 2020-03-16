import numpy as np
import math

def get_minibatches(X, Y, minibatch_size):
    m = X.shape[1]
    num_complete_minibatches = math.floor(m / minibatch_size)
    minibatches = []
    for i in range(num_complete_minibatches):
        minibatch_X = X[:, i*minibatch_size:(i+1)*minibatch_size]
        minibatch_Y = Y[:, i*minibatch_size:(i+1)*minibatch_size]
        minibatch = (minibatch_X, minibatch_Y)
        minibatches.append(minibatch)

    if m % minibatch_size != 0:
        minibatch_X = X[:, num_complete_minibatches * minibatch_size:]
        minibatch_Y = Y[:, num_complete_minibatches * minibatch_size:]
        minibatch = (minibatch_X, minibatch_Y)
        minibatches.append(minibatch)

    return minibatches
