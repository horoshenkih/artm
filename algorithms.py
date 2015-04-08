import numpy as np

def nmf(collection, t, phi, theta, n_iter=10, algorithm='em',params=None,verbose=False):
    w, d = collection.shape
    losses_sliding_window = []
    sliding_window_size = 3
    for iteration in range(n_iter):
        if verbose:
            print "Iteration", iteration
        n_wt = np.zeros((w,t))
        n_dt = np.zeros((d,t))
        product = phi * theta
        if verbose:
            print "All triples"
        for d_i in range(d):
            for w_i in range(w):
                Z = product[w_i,d_i]
                for t_i in range(t):
                    delta = collection[w_i,d_i]*phi[w_i,t_i]*theta[t_i,d_i]/Z
                    n_wt[w_i,t_i] += delta
                    n_dt[d_i,t_i] += delta

        if verbose:
            print "Fill Phi"
        for t_i in range(t):
            n_t = 0.
            if algorithm == 'em':
                for w_i in range(w):
                    phi[w_i,t_i] = n_wt[w_i,t_i]
                    n_t += n_wt[w_i,t_i]
            elif algorithm == 'lda':
                beta = params['beta']
                for w_i in range(w):
                    increment = n_wt[w_i,t_i] + beta[w_i]
                    phi[w_i,t_i] = increment
                    n_t += increment
            else:
                print "not implemented"
                return
            # normalize
            for w_i in range(w):
                phi[w_i,t_i] /= n_t
        if verbose:
            print "Fill Theta"
        for d_i in range(d):
            n_d = 0.
            if algorithm == 'em':
                for t_i in range(t):
                    theta[t_i,d_i] = n_dt[d_i,t_i]
                    n_d += n_dt[d_i,t_i]
            elif algorithm == 'lda':
                alpha = params['alpha']
                for t_i in range(t):
                    increment = n_dt[d_i,t_i] + alpha[t_i]
                    theta[t_i,d_i] = increment
                    n_d += increment
            else:
                print "not implemented"
                return
            # normalize
            for t_i in range(t):
                theta[t_i,d_i] /= n_d
        # find losses
        L = 0.
        for d_i in range(d):
            for w_i in range(w):
                for t_i in range(t):
                    L += collection[w_i, d_i] * np.log(phi[w_i, t_i]*theta[t_i, d_i])
        #L /= w * t * d
        loss_phi = 0.
        beta = params['beta']
        for w_i in range(w):
            for t_i in range(t):
                loss_phi += beta[w_i]*np.log(phi[w_i, t_i])
        #loss_phi /= w * t
        loss_theta = 0.
        alpha = params['alpha']
        for d_i in range(d):
            for t_i in range(t):
                loss_theta += alpha[t_i]*np.log(theta[t_i, d_i])
        #loss_theta /= d * t
        if verbose:
            print "Likelihood:", L
            print "Loss phi:", loss_phi
            print "Loss theta:", loss_theta
        # collect sliding window
        current_loss = L + loss_theta + loss_phi
        if iteration < 2 * sliding_window_size:
            losses_sliding_window.append(current_loss)
        else:
            # split window and calc errors
            early, later = losses_sliding_window[:5], losses_sliding_window[5:]
            early_loss = sum(early) / len(early)
            later_loss = sum(later) / len(later)
            if verbose:
                print "early stopping data"
                print "early loss:", early_loss
                print "later loss:", later_loss
                print "/ early stopping data"
            if later_loss < early_loss:
                return phi, theta
            # add new loss
            losses_sliding_window.pop(0)
            losses_sliding_window.append(current_loss)
    return phi, theta
