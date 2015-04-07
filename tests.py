from utils import *
from algorithms import nmf
import numpy as np

w = 500
d = 200
t = 10

for alpha0 in [0.1, 0.5, 1., 2., 5., 10.]:
    beta0 = 0.1  # const
    alpha = np.ones((1,t)).ravel() * alpha0
    beta = np.ones((1,w)).ravel() * beta0

    phi, theta, prod0, nd, collection = generate_all(w,d,t,alpha,beta,seed=31)

    seed=42
    phi0 = generate_phi(w,t,beta,seed=seed)
    theta0 = generate_theta(d,t,alpha,seed=seed)

    '''
    d_phi0 = dist(phi, phi0)
    d_theta0 = dist(theta, theta0)
    d_prod0 = dist (prod0, phi0 * theta0)
    '''

    algorithm = 'lda'
    n_iter = 50
    params = {
        'alpha': alpha,
        'beta': beta,
    }

    phi_em, theta_em = nmf(collection, t, phi0, theta0, algorithm=algorithm, n_iter=n_iter, params=params)

    '''
    print "Initial distances:"
    print d_phi0
    print d_theta0
    print d_prod0
    '''

    print "Alpha0:", alpha0
    print "D(phi, phi0):", dist(phi, phi_em)
    print "D(theta, theta0)", dist(theta, theta_em)
    print "D(prod, prod0)", dist (prod0, phi_em * theta_em)
