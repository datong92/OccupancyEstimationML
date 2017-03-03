'''
Implements classes and methods to estimate occupancy
patterns from energy consumption data.

@date: Apr 16, 2016, Last Updated: Mar 2, 2017.
@author: Datong Paul Zhou
'''
import pickle
import copy as cp
import numpy as np

from src.Estimation import *
from sklearn.linear_model import LinearRegression


def main():
    user_num = 3  # User Number: [0, 24]

    X, Y = pickle.load(open("Data/User{}.p".format(user_num), "rb"))

    fitted_model = LinearRegression(fit_intercept=False).fit(X, Y)

    pert_initial_w = np.tile(fitted_model.coef_.reshape(-1, 1), (1, 2))
    pert_initial_w += 0.003*np.random.normal(size=pert_initial_w.shape)

    # Fit Conditional Mixture Model
    num_train_pts = 2500
    mix = ConditionalMixtureModel(initial_w=pert_initial_w,
                                  initial_pi=0.5*np.ones(2), initial_var=0.1,
                                  num_mixt=2, X=X[:num_train_pts],
                                  Y=Y.flatten()[:num_train_pts])
    mix.TrainModel(500)

    return


if '__name__' == '__main__':
    main()
