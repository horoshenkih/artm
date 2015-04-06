import numpy as np

def em(collection, t, phi, theta, n_iter=10):
    w, d = collection.shape
    for iteration in range(n_iter):
        n_wt = np.zeros((w,t))
        n_dt = np.zeros((d,t))
        product = phi * theta
        for d_i in range(d):
            for w_i in range(w):
                Z = product[w_i,d_i]
                #print w_i, d_i, Z
                for t_i in range(t):
                    delta = collection[w_i, d_i]*phi[w_i,t_i]*theta[t_i,d_i]/Z
                    n_wt[w_i,t_i] += delta
                    n_dt[d_i,t_i] += delta
        for t_i in range(t):
            n_t = n_wt[:,t_i].sum()
            for w_i in range(w):
                phi[w_i,t_i] = n_wt[w_i,t_i] / n_t
        for d_i in range(d):
            n_d = n_dt[d_i,:].sum()
            for t_i in range(t):
                theta[t_i,d_i] = n_dt[d_i,t_i] / n_d
    return phi, theta

def lda(collection, t, phi, theta):
    # TODO
    return phi, theta
