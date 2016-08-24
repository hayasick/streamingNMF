from __future__ import division
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import zeta as spzeta

def zeta(alpha, xmin, xmax=float('Inf')):
    return spzeta(alpha, xmin)


#def zeta(alpha, xmin, xmax=float('Inf')):
#    if xmax == float('Inf'):
#        return spzeta(alpha, xmin)
#    else:
#        return np.sum((np.arange(xmax) + xmin) ** -alpha)

def distribution(x, alpha, xmin=1):
    """
    x takes integer
    """
    return (x ** -alpha) / zeta(alpha, xmin, x.size)

def CDF(x_rank, alpha, xmin=1):
    """
    x_rank takes integer
    """
    return np.array([np.sum(distribution(x_rank, alpha, xmin)[:i]) for i in xrange(x_rank.size)])

def X_CDF(x, x_rank, xmin=1):
    x_trimmed = x[(xmin - 1):]
    N = np.sum(x_trimmed)

    return np.array([np.sum(x_trimmed[:i]) / N for i in x_rank])


def loglikelihood(x, alpha, xmin=1, summed=True):
    """
    log-likelihood of powerlaw distribution (discrete).
    x is expected as sorted by decreasing order.
    xmin takes 1 <= xmin <= len(x).
    """
    x_trimmed = x[(xmin - 1):]
    x_rank = np.arange(xmin, x.size + 1)

#    N = np.sum(x_trimmed)
#    llh = - N * np.log(zeta(alpha, xmin)) - alpha * np.sum(x_trimmed * np.log(x_rank))
    llh = - x_trimmed * (np.log(zeta(alpha, xmin, x.size)) + alpha * np.log(x_rank))
    if summed:
        llh = np.sum(llh)
    return llh

def get_scaleparam(x, xmin):
    """
    estimate scale parameter alpha by maximizing discrete power law likelihood
    """
    f = lambda a: -loglikelihood(x, a, xmin)
    #res = minimize_scalar(f, bounds=(1, 3), method='bounded')
    res = minimize_scalar(f, bounds=(0, 4), method='bounded')
    return res.x

def get_KS(x, alpha, xmin=1):
    """
    compute Kolmogorov-Sminov distance
    """
    x_rank = np.arange(xmin, x.size + 1)
    return np.max(np.abs(X_CDF(x, x_rank, xmin) - CDF(x_rank, alpha, xmin)))

def get_xmin(x, xmin_max=70):
    KS = np.zeros(xmin_max - 1)
    for xmin in xrange(1, xmin_max):
        alpha = get_scaleparam(x, xmin)
        #print alpha,
        KS[xmin - 1] = get_KS(x, alpha, xmin)
    return np.argmin(KS) + 1

if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    
    U = np.loadtxt(sys.argv[1])
    K = U.shape[1]
    xmins = list()
    for k in xrange(K):
        xmins.append(get_xmin(U[:, k]))
        print k, xmins[k]
        


    U_normalized = U / np.sum(U, axis=0)
    x = range(1, U.shape[0] + 1)
    for k in xrange(K):
        #plt.subplot(k % 5 + 1, int(k / 5) + 1, k + 1)
        plt.subplot(4, 3, k + 1)
        plt.title('Topic %d' % (k + 1))
        plt.loglog(x, U_normalized[:, k])
        plt.loglog(x, distribution(x, get_scaleparam(U_normalized[:, k], xmins[k]), xmins[k]))
        plt.axvline(xmins[k], ls='dashed')
        #plt.loglog(x, distribution(x, get_scaleparam(U_normalized[:, k], 10), 10))
        #plt.loglog(x, U[:, k])
    #plt.legend(range(k))
    plt.show()
