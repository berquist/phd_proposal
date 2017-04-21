from __future__ import print_function

import os
import pickle
import sys
import numpy as np
import scipy
import scipy.io


# --------------------------------------------
# Parameters
# --------------------------------------------
split = int(sys.argv[1]) # test split for cross-validation (between 0 and 5)

# --------------------------------------------
# Load data and models
# --------------------------------------------
if not os.path.exists('qm7.mat'):
    os.system('wget http://www.quantum-machine.org/data/qm7.mat')
dataset = scipy.io.loadmat('qm7.mat')
with open('nn-%d.pkl' % split) as fh:
    nn = pickle.load(fh)

print('results after %d iterations'%nn.nbiter)

Ptrain = dataset['P'][range(0, split) + range(split + 1, 5)].flatten()
Ptest = dataset['P'][split]

for P, name in zip([Ptrain, Ptest], ['training', 'test']):
    # --------------------------------------------
    # Extract test data
    # --------------------------------------------
    X = dataset['X'][P]
    T = dataset['T'][0, P]

    # --------------------------------------------
    # Test the neural network
    # --------------------------------------------
    print('\n%s set:' % name)
    Y = np.array([nn.forward(X) for _ in range(10)]).mean(axis=0)
    print('MAE:  %5.2f kcal/mol' % np.abs(Y-T).mean(axis=0))
    print('RMSE: %5.2f kcal/mol' % np.square(Y-T).mean(axis=0) ** 0.5)
