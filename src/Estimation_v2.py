'''
Class definitions of Hidden Markov Model (HMM) and Conditional Gaussian
Mixture Model (CGMM) for estimating modes from hourly
smart meter consumption data

@date: Apr 14, 2016, Updated: Aug 31, 2017.
@author: Datong Paul Zhou
'''

import math
import copy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime

from collections import namedtuple
from scipy.stats import expon, poisson

from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, LinearRegression

from sklearn.svm import SVR
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.grid_search import RandomizedSearchCV
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV

# Suppress warnings
pd.options.mode.chained_assignment = None


def random_data(length, mc_trans_matrix, num_mix_comp,
                emission_statistics, nfact=0.1):
    """
    Generate random Gaussian sequence with desired number of MC states and
    mixture components. State Transition Probabilities are inputs, but the
    mixture component indicator probabilities are randomly set.
    """

    # Generate random mixture component indicator matrix
    num_mc_states = mc_trans_matrix.shape[0]
    mixture = np.random.uniform(0, 1, [num_mc_states, num_mix_comp])
    for j in range(num_mc_states):
        mixture[j] /= sum(mixture[j])

    # Generate random numbers in [0, 1]
    randnumbers = np.random.rand(length-1)
    randnumbers2 = np.random.rand(length)

    def next_state(prob_matrix, state, randnum):
        nextstate = 0
        count = 0.0
        for j in range(prob_matrix.shape[1]):
            count += prob_matrix[state, j]
            if randnum < count:
                break
            nextstate += 1
        return nextstate

    # Generate random sample path of hidden states
    mc_states = [0]
    for j in range(length-1):
        mc_states.append(next_state(mc_trans_matrix,
                         mc_states[j], randnumbers[j]))
    mc_states = np.array(mc_states)

    # Generate random mixture components and corresponding emissions
    mix_states = []
    observations = []
    for j in range(len(mc_states)):
        mix_states.append(next_state(mixture, mc_states[j], randnumbers2[j]))
        observations.append(np.random.normal(emission_statistics[0, mc_states[j], mix_states[j]],
                            nfact*np.sqrt(emission_statistics[1, mc_states[j], mix_states[j]])+1e-12))
    mix_states = np.array(mix_states)
    observations = np.array(observations)

    return mixture, observations, mc_states


def cleandata(data, dr_times, standardize_air_temp=True, normalize_kWh=True):
    """
    Only choose entire days, eliminate days with missing hours.
    Updates DR Times.
    If standardize_cols=True (default=True), then standardize kWh and air_temp
    """
    start_locs = [data.index.get_loc(data.index[i]) for i in range(len(data.index)) if data.index.hour[i] == 0]
    start_locs = [start_locs[i] for i in range(len(start_locs)-1) if ((start_locs[i+1]-start_locs[i])==24)]
    keep_locs = np.concatenate([np.array(range(start_locs[i], start_locs[i]+24)) for i in range(len(start_locs))])

    # Clean Data
    cleaned_data = data.iloc[keep_locs, :]

    # Normalize kWh and standardize air_temp
    if standardize_air_temp:
        cleaned_data['air_temp'] = (cleaned_data['air_temp'] - cleaned_data['air_temp'].mean()) / cleaned_data['air_temp'].std()
    if normalize_kWh:
        cleaned_data['kWh'] = (cleaned_data['kWh'] - cleaned_data['kWh'].min())
        / (cleaned_data['kWh'].max() - cleaned_data['kWh'].min())

    # Update DR Times
    dr_times_cleaned = [dr_times[i] for i in range(len(dr_times)) if dr_times[i] in cleaned_data.index.astype(pd.Timestamp)]

    return cleaned_data, dr_times_cleaned


def mape_point(ground_truth, prediction):
    """
    Computes pointwise MAPE.
    Warning: No threshold, no abortion even in case of negative values
    """
    mape = 100.0 * np.abs((ground_truth - prediction) / ground_truth)
    return mape


class Gaussian:
    """
    Gaussian Emissions: PDF and Update Equation for HMM Parameters
    """
    def __init__(self, params):
        self.mu = params[0]
        self.var = params[1]

    def P(self, k):
        prob = 1.0 / np.sqrt(2 * pi * self.var) * np.exp(-(k - self.mu)**2.0 / (2.0 * self.var))
        return prob

    def ML(self, data, weights):
        self.mu = 1.0*sum(data * weights) / sum(weights)
        self.var = 1.0*sum(weights * (data - self.mu)**2) / sum(weights)


class Poiss:
    """
    Poissonian Emissions: PDF and Update Equation for HMM Parameters
    """
    def __init__(self, param):
        self.param = param

    def P(self, k):
        def logfac(n):
            return sum(np.log(range(1, n+1)))
        log_p = k * np.log(self.param) - self.param - logfac(k)
        return np.exp(log_p)

    def ML(self, data, weights):
        self.param = sum(data * weights) / sum(weights)


def DeepCopy(array):
    return cp.deepcopy(array)


class HiddenMM:
    """
    - Initial Distribution of Markov Chain (MC)
    - MC transition probabilities
    - Mixture transition probabilities
    - Matrix of exponential families carrying MC states and mixture components
    - Matrix of State Transition Constraints
    """
    def __init__(self, initial_prob, mc_transition_prob,
                 mix_transition_prob, families, mc_constraints):
        self.initial = cp.copy(initial_prob)
        self.mc_transition = cp.copy(mc_transition_prob)
        self.mix_transition = cp.copy(mix_transition_prob)
        self.families = DeepCopy(families)
        self.mc_constraints = cp.copy(mc_constraints)

        self.alpha = None
        self.beta = None

        n_mc_states = initial_prob.shape[0]
        n_mix_states = mix_transition_prob.shape[1]

        self.mc_states = range(n_mc_states)
        self.mix_states = range(n_mix_states)

        assert(n_mc_states == self.mc_transition.shape[0])
        assert(n_mc_states == self.mc_transition.shape[1])
        assert(n_mc_states == self.mix_transition.shape[0])
        assert(n_mc_states == self.families.shape[0])
        assert(n_mix_states == self.families.shape[1])

    def FullObservationProbabilities(self, datum):
        """
        Returns a 2d array of conditional probabilities P(y = datum | q = i, p = j)
        for all states i in the Markov chain and mixture components j.
        Hidden state q, mixture component indicator p
        """
        ret = [i.P(datum) for i in self.families.flatten()]
        return np.array(ret).reshape((len(self.mc_states), len(self.mix_states)))

    def ObservationProbabilities(self, datum):
        """
        Returns an array of conditional probabilities P(y = datum | q = i)
        for all states i in the Markov chain.
        """
        # Get Full Observation Probabilities
        full_probs = self.FullObservationProbabilities(datum)

        # Marginalize over the mixture components
        ret = np.zeros(len(self.mc_states))
        for i in self.mc_states:
            ret[i] = np.dot(full_probs[i, :], self.mix_transition[i, :])
        return ret

    def AlphaRecursion(self, data):
        """
        Perform alpha-recursion with given data. Normalized.
        Returns a [ndata x nstates] probability matrix, whose (t,i) element is
        (up to a constant depending on the data) p(y_1, ..., y_{t-1}, q_t = i)
        """
        alpha = np.zeros((len(data), len(self.mc_states)))

        # Initialize with initial probability in state 1.
        # Initial state is given, so p(y_0 | q_0) = 1
        alpha[0, :] = self.initial
        log_normalization = np.log(sum(alpha[0, :]))
        alpha[0, :] /= sum(alpha[0, :])

        # Loop through all times 1, ..., t-1
        for t in range(1, len(data)):
            data_prob = self.ObservationProbabilities(data[t-1])
            alpha[t, :] = np.dot(self.mc_transition.T, alpha[t-1, :] * data_prob)
            log_normalization += np.log(sum(alpha[t, :]))
            alpha[t, :] /= sum(alpha[t, :])

        # p(y) = \sum_i p(y, q_t = i) = \sum_i alpha_t(i) p(y_t | q_t)
        # Note that this is different from the formula in the book since we've
        # modified alpha to be p(y_1, ..., y_{t-1}, q_t) instead of
        # p(y_1, ..., y_t, q_t). Also, our alpha is normalized.
        data_prob = self.ObservationProbabilities(data[-1])
        self.alpha = alpha
        self.log_likelihood = log_normalization + np.log(sum(alpha[t, :] * data_prob))
        return alpha

    def BetaRecursion(self, data):
        """
        Perform normalized beta-recursion with given data.
        Return probability matrix.
        """
        beta = np.zeros((len(data), len(self.mc_states)))
        beta[-1, :] = np.ones((1, len(self.mc_states))) / len(self.mc_states)
        for t in range(len(data)-2, -1, -1):
            data_prob = self.ObservationProbabilities(data[t+1])
            beta[t, :] = np.dot(self.mc_transition, (beta[t+1, :] * data_prob))
            beta[t, :] /= sum(beta[t, :])
        self.beta = beta
        return beta

    def LogLikelihood(self, data):
        self.AlphaRecursion(data)
        return self.log_likelihood

    def DoInference(self, data):
        """
        Finds the probabilities
        1) phi_t(i, j) = P(q_t = i, p_t = j | y)            --> m_ij
        2) xi_{t,t+1}(i, j) = P(q_{t+1} = j, q_t = i | y)   --> n_ij
        Returns a pair: (3D phi array, 3D xi array)
        """
        alpha = self.AlphaRecursion(data)
        beta = self.BetaRecursion(data)
        phi = np.zeros((len(data), len(self.mc_states), len(self.mix_states)))
        xi = np.zeros((len(data)-1, len(self.mc_states), len(self.mc_states)))
        for t in range(len(data)):

            # phi
            data_prob = self.ObservationProbabilities(data[t])
            full_data_prob = self.FullObservationProbabilities(data[t])
            phi[t, :] = np.array(self.mix_transition) * full_data_prob * \
            np.array(alpha[t, :] * beta[t, :], ndmin=2).T
            # Divide by p(y) (= normalization) to obtain m_ij
            phi[t, :] /= sum(phi[t, :])

            # xi
            if t < len(data) - 1:
                next_data_prob = self.ObservationProbabilities(data[t+1])
                # alpha was defined as p(y_1, ..., y_{t-1}, q_t) instead of p(y_1, ..., y_t, q_t):
                # --> Must multiply data_prob to alpha[t,:]
                xi[t, :] = np.outer(alpha[t, :] * data_prob, beta[t+1, :] *
                                    next_data_prob) * np.array(self.mc_transition)

                # Apply MC state transition matrix constraints:
                xi = xi * self.mc_constraints

                # Divide by p(y) (= normalization) to obtain n_ij
                xi[t, :] /= sum(xi[t, :])
        return (phi, xi)

    def UpdateParameters(self, data):
        """
        Given some data, updates the current parameters (MC transition matrix,
        sufficient statistics of emission distributions) by taking one step
        under the EM algorithm.
        """
        (mij, nij) = self.DoInference(data)

        self.mc_transition = sum(nij, axis=0)
        self.mix_transition = sum(mij, axis=0)
        for i in self.mc_states:
            self.mc_transition[i, :] /= sum(self.mc_transition[i, :])
            self.mix_transition[i, :] /= sum(self.mix_transition[i, :])
            for j in self.mix_states:
                self.families[i, j].ML(data, mij[:, i, j])

    def Predict(self, data):
        """
        After the parameters have been learned, solve the prediction problem:
        P(Q(t+1) | y_0, ..., y_t).
        Make use of recursive formula for alpha(q_t+1) = fn(alpha(q_t))
        Returns an array of dimension (num_mc_states, 1)
        """
        data_prob = self.ObservationProbabilities(data[-1])
        alpha_t1 = np.dot(self.mc_transition.T, self.alpha[-1, :] * data_prob)
        alpha_t1 /= sum(alpha_t1)
        return alpha_t1

    def Smooth(self, idx, data):
        """
        After the parameters have been learned, solve the smoothing problem:
        P(Q(idx) | y_0, ..., y_t), where 1 <= idx <= t-1
        Returns an array of dimension (num_mc_states, 1)
        """
        assert((idx >= 1) and (idx <= len(data)-2))
        data_prob = self.ObservationProbabilities(data[idx])
        product = self.alpha[idx, :] * data_prob * self.beta[idx, :]
        product /= sum(product)
        return product

    def Filter(self, data):
        """
        After the parameters have been learned, solve the filtering problem:
        P(Q(t) | y_0, ..., y_t).
        Returns an array of dimension (num_mc_states, 1)
        """
        data_prob = self.ObservationProbabilities(data[-1])
        product = self.alpha[-1, :] * data_prob
        product /= sum(product)
        return product

    def PredictionAccuracy(self, mc_truth, mix_truth, emission_stats_truth):
        """
        Compute the (component-wise) MAPE of estimated
        a) Markov Chain Transition Matrix
        b) Mixture Component Indicator Matrix
        c) Emission Statistics
        """
        mape_mc = (self.mc_transition - mc_truth) / mc_truth * 100.0
        mape_mix = (self.mix_transition - mix_truth) / mix_truth * 100.0

        mus_estimated = []
        vars_estimated = []
        for i in range(self.families.shape[0]):
            for j in range(self.families.shape[1]):
                mus_estimated.append(self.families[i, j].mu)
                vars_estimated.append(self.families[i, j].var)
        mus_estimated = np.array(mus_estimated).reshape(self.families.shape)
        vars_estimated = np.array(vars_estimated).reshape(self.families.shape)

        mape_mus = (mus_estimated - emission_stats_truth[0]) / emission_stats_truth[0] * 100.0
        mape_vars = (vars_estimated - emission_stats_truth[1]) / emission_stats_truth[1] * 100.0

        return mape_mc, mape_mix, mape_mus, mape_vars


class ConditionalMixtureModel:
    """
        Conditional Mixture Model:
        - Assumes same covariance for all classes (can be changed later)
        - Assumes input format of w = [w_1, ..., w_n] (column vectors)
    """
    def __init__(self, initial_w, initial_pi, initial_var, num_mixt, X, Y):
        self.w = cp.copy(initial_w)
        self.pi = cp.copy(initial_pi)
        self.var = cp.copy(initial_var)
        self.num_mixt = num_mixt
        self.X = cp.copy(X)
        self.Y = cp.copy(Y)
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
        for it in tqdm(range(num_iter)):
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
            Do Pointwise DR Prediction to compute counterfactual.
            Takes in training data X to compute predictions in 3 different ways
        """

        clsest_ix = np.argmin(np.linalg.norm(self.X[:ntrpts] - Xpred, axis=1))
        latent_var = np.argmax(self.probas[clsest_ix])
        preds1 = np.dot(Xpred, self.w[:, latent_var])
        preds2 = np.sum(np.dot(Xpred, self.w) * self.probas[clsest_ix])
        preds3 = ord_model.predict(Xpred.reshape(1, -1))[0, 0]

        return preds1, preds2, preds3
