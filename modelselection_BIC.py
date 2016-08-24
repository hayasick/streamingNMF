from __future__ import division
import numpy as np
from modelselection import *

def get_pis(x, gap):
    x_normalized = x / np.sum(x)
    pi  = np.mean(x_normalized[:gap])
    _pi = np.mean(x_normalized[gap:])
    assert np.abs(pi * gap + _pi * (x.size - gap) - 1) < 1e-10, 'sum of probs must be 1'
    return (pi, _pi)

def loglikelihood_step(x, gap, pis=None, summed=True):
    assert gap >= 1, 'gap must be >= 1'
    if pis is None:
        pis = get_pis(x, gap)
        
    pi_vec = np.array([pis[0]] * gap + [pis[1]] * (x.size - gap))
    llh = x * np.log(pi_vec)

    if summed:
        llh = np.sum(llh)
    return llh

def get_LR(x, xmin_max=None):
    """
    compute log-likelihood ratio
    note that xmin of powerlaw is assume to be 0 to compute proper likelihood ratio
    """
    N = np.sum(x)
    x_normalized = x / N
    if xmin_max is None:
        xmin_max = x.size - 1
    
    xmin = 1#get_xmin(x, xmin_max=70)
    alpha = get_scaleparam(x, xmin)
    power = loglikelihood(x_normalized, alpha, xmin, summed=False)

    # adjustment term to absorb the difference of parameter dimension
    power_adj = - 0.5 * np.log(N)
    #step_adj  = - 0.5 * 2 * np.log(N)
    step_adj  = - 0.5 * np.log(N)

    LR = np.zeros(xmin_max)
    for gap in xrange(1, xmin_max + 1):
        step = loglikelihood_step(x_normalized, gap, summed=False)
        diff = step - power
        diff_adj  = step_adj - power_adj

        omega2 =   np.sum(x * diff ** 2) / N \
                 - np.sum(x * diff / N) ** 2
        LR[gap - 1] = (np.sum(x * diff) + diff_adj) / np.sqrt(N * omega2)
        
    return LR
        
def get_BIC(x, xmin_max=None):
    """
    compute BIC of powerlaw and 2-step distribution
    """

    if xmin_max is None:
        xmin_max = x.size
    N = np.sum(x)
    
    BICs = np.zeros(xmin_max)
    xmin = get_xmin(x, xmin_max=70)
    alpha = get_scaleparam(x, xmin)
    BICs[0] = loglikelihood(x, alpha, xmin) - 0.5 * np.log(N)
    for gap in xrange(1, xmin_max):
        BICs[gap] = loglikelihood_step(x, gap) - 0.5 * 2 * np.log(N)
    return BICs

if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    
    U = np.loadtxt(sys.argv[1])
    K = U.shape[1]
    
    #for k in xrange(K):
    #    bic = get_BIC(U[:, k])
    #    plt.plot(bic)
    #    print (k + 1, np.argmax(bic))

    sd = 1.96
    for k in xrange(K):
        LR = get_LR(U[:, k])
        plt.plot(LR)
        print (k + 1, np.where(LR > sd))

    plt.xlim(-1, 20)
    plt.show()
