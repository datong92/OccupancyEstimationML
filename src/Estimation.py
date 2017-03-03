'''
Implements classes and methods to estimate occupancy
patterns from energy consumption data.

@date: Apr 16, 2016, Last Updated: Mar 2, 2017.
@author: Datong Paul Zhou
'''

import copy as cp
import numpy as np


class ConditionalMixtureModel:
    """
        Conditional Mixture Model:
        - Assumes same covariance for all classes (can be changed later)
        - Assumes input format of w = [w_1, ..., w_n] (column vectors)
    """
    def __init__(self, initial_w, initial_pi, initial_var, num_mixt, X, Y):
        self.w = cp.deepcopy(initial_w)
        self.pi = cp.deepcopy(initial_pi)
        self.var = cp.deepcopy(initial_var)
        self.num_mixt = num_mixt
        self.X = cp.deepcopy(X)
        self.Y = cp.deepcopy(Y)
        self.probas = None

        assert(self.X.shape[1] == self.w.shape[0]), 'Number of columns and length of w disagree!'
        assert(self.X.shape[0] == len(self.Y)), 'Dimension of X and Y disagree!'
        assert(self.num_mixt == len(self.pi)), 'Dimension of initial states is wrong!'

    def EStep(self):
        """
            Does the E-Step of the EM-Algorithm
            Returns [n x k] matrix of posterior probabilities
        """
        gamma = np.zeros([len(self.Y), self.num_mixt])
        for k in range(self.num_mixt):
            probas = np.exp(-(self.Y.flatten() - np.inner(self.X,
                              self.w[:, k]))**2 /
                             (2*self.var)) / np.sqrt(self.var)
            gamma[:, k] = (self.pi[k] * probas)
        gamma /= np.tile(np.sum(gamma, axis=1).reshape(-1, 1), self.num_mixt)
        self.probas = gamma
        return gamma

    def MStep(self):
        """
            Does the M-Step of the EM-Algorithm
            - Updates pi_1, ..., pi_k
            - Updates the weights.
        """

        gamma = self.EStep()

        # Update mixture proportions
        self.pi = np.sum(gamma, axis=0) / len(gamma)

        # Update weights
        for k in range(self.num_mixt):
            XTD = np.dot(self.X.T, np.diag(gamma[:, k]))
            self.w[:, k] = np.dot(np.linalg.inv(np.dot(XTD, self.X)),
                                  np.dot(XTD, self.Y))

        # Update variance
        var = 0.0
        for k in range(self.num_mixt):
            var += np.dot(gamma[:, k], (self.Y.flatten() -
                          np.dot(self.X, self.w[:, k]))**2)
        self.var = var / len(gamma)
        return

    def LogLikelihood(self, gamma):
        """
            Compute Log Likelihood of Data
        """
        return np.sum(np.sum(gamma * np.log(self.probas)))

    def TrainModel(self, num_iter):
        """
            Do E-Step, then M-Step
        """
        for it in range(num_iter):
            gamma = self.EStep()
            self.MStep()
            if it % 100 == 0:
                print('Log Likelihood after {} iterations is {}'.format(it,
                      self.LogLikelihood(gamma)))
        return

    def TrainError(self, losstype='MAPE'):
        """
            Calculate training error. Type is either 'MAPE' or 'SquaredLoss'
        """
        assert ((losstype == 'MAPE') or (losstype == 'SquaredLoss')), 'Unknown Loss Type'

        preds = np.sum(np.dot(self.X, self.w) * self.probas, axis=1)
        if losstype == 'MAPE':
            return 100.0 * np.abs((preds - self.Y) / self.Y)
        else:
            return (preds - self.Y)**2

    def StepwisePrediction(self, nsteps, ntrpts, X, Y, ord_model):
        """
            Do Stepwise Forward Prediction with Mixture Model in 2 ways:
            1) Find closest neighbor, use hard 0 / 1 (rounded)
            2) Find closest neighbor, use posterior probabilities
            3) Ordinary Model
        """
        clsest_ix = [np.argmin(np.linalg.norm(self.X[:ntrpts] - X[ntrpts+it],
                     axis=1)) for it in range(nsteps)]
        latent_var = np.array([np.argmax(self.probas[cl]) for cl in clsest_ix])
        preds1 = np.array([np.dot(X[ntrpts+it], self.w[:, latent_var[it]])
                          for it in range(nsteps)])
        preds2 = np.array([np.sum(np.dot(X[ntrpts+it], self.w) *
                          self.probas[clsest_ix[it]]) for it in range(nsteps)])
        preds3 = ord_model.predict(X[ntrpts:ntrpts + nsteps])
        return preds1, preds2, preds3

    def PointwisePrediction(self, ntrpts, dr_tstamp, X, Xpred, ord_model):
        """
            Do Pointwise Prediction.
            Takes in training data X to compute predictions in 3 different ways
        """

        clsest_ix = np.argmin(np.linalg.norm(self.X[:ntrpts] - Xpred, axis=1))
        latent_var = np.argmax(self.probas[clsest_ix])
        preds1 = np.dot(Xpred, self.w[:, latent_var])
        preds2 = np.sum(np.dot(Xpred, self.w) * self.probas[clsest_ix])
        preds3 = ord_model.predict(Xpred.reshape(1, -1))[0, 0]

        return preds1, preds2, preds3
