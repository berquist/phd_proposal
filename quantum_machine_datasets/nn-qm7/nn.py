import numpy as np


class Module(object):

    def forward(self, X):
        pass

    def backward(self, DY):
        pass

    def update(self, lr):
        pass

    def average(self, nn, a):
        pass


class Sequential(Module):

    def __init__(self, modules):
        self.modules = modules

    def forward(self, X):
        for m in self.modules:
            X = m.forward(X)
        return X

    def backward(self, DY):
        for m in self.modules[::-1]:
            DY = m.backward(DY)
        return DY

    def update(self, lr):
        for m in self.modules:
            X = m.update(lr)

    def average(self, nn, a):
        for m, n in zip(self.modules, nn.modules):
            m.average(n, a)


class Input(Module):

    def __init__(self, X):
        self.step = 1.0
        self.noise = 1.0
        self.triuind = (np.arange(23)[:, np.newaxis] <= np.arange(23)[np.newaxis, :]).flatten()
        self.max = 0
        for _ in range(10):
            self.max = np.maximum(self.max, self.realize(X).max(axis=0))
        X = self.expand(self.realize(X))
        self.nbout = X.shape[1]
        self.mean = X.mean(axis=0)
        self.std = (X - self.mean).std()

    def realize(self, X):
        def _realize_(x):
            inds = np.argsort(-(x**2).sum(axis=0)**0.5 + np.random.normal(0, self.noise, x[0].shape))
            x = x[inds, :][:, inds] * 1
            x = x.flatten()[self.triuind]
            return x
        return np.array([_realize_(z) for z in X])

    def expand(self, X):
        Xexp = []
        for i in range(X.shape[1]):
            for k in np.arange(0, self.max[i] + self.step, self.step):
                Xexp += [np.tanh((X[:, i] - k) / self.step)]
	# pylint: disable=no-member
        return np.asarray(Xexp).T

    def normalize(self, X):
        return (X - self.mean) / self.std

    def forward(self, X):
        return self.normalize(self.expand(self.realize(X))).astype('float32')


class Output(Module):

    def __init__(self, T):
        self.tmean = T.mean()
        self.tstd = T.std()
        self.nbinp = 1

    def forward(self, X):
        self.X = X.flatten()
        return self.X * self.tstd + self.tmean

    def backward(self, DY):
        return (DY / self.tstd).astype('float32')[:, np.newaxis]


class Linear(Module):

    def __init__(self, m, n):

        self.tr = m ** 0.5 / n ** 0.5
        self.lr = 1 / m ** 0.5

        self.W = np.random.normal(0, 1 / m ** 0.5, [m, n]).astype('float32')
        self.A = np.zeros([m]).astype('float32')
        self.B = np.zeros([n]).astype('float32')

    def forward(self, X):
        self.X = X
        Y = np.dot(X - self.A, self.W) + self.B
        return Y

    def backward(self, DY):
        self.DW = np.dot((self.X - self.A).T, DY)
        self.DA = -(self.X - self.A).sum(axis=0)
        self.DB = DY.sum(axis=0) + np.dot(self.DA, self.W)
        DX = self.tr * np.dot(DY, self.W.T)
        return DX

    def update(self, lr):
        self.W -= lr * self.lr * self.DW
        self.B -= lr * self.lr * self.DB
        self.A -= lr * self.lr * self.DA

    def average(self, nn, a):
        self.W = a * nn.W + (1 - a) * self.W
        self.B = a * nn.B + (1 - a) * self.B
        self.A = a * nn.A + (1 - a) * self.A


class Sigmoid(Module):

    def forward(self, X):
        self.Y = np.tanh(X / 1.5)
        return 1.5 * self.Y

    def backward(self, DY):
        return DY * (1 - self.Y ** 2)

