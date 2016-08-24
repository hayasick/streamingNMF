import numpy as np
import codecs
import mapXm
import sys
from collections import defaultdict
import datetime


def to_seconds_float(timedelta):
    """Calculate floating point representation of combined
    seconds/microseconds attributes in :param:`timedelta`.

    :raise ValueError: If :param:`timedelta.days` is truthy.

        >>> to_seconds_float(datetime.timedelta(seconds=1, milliseconds=500))
        1.5
    """
    return timedelta.seconds + timedelta.microseconds / 1E6 \
        + timedelta.days * 86400


class FM_base(object):
    """
    Base class of factorization methods.
    """
    def __init__(self, result_pref=''):

        self.mapper = mapXm.Map_Xm()
        self.err = list()
        self.U = list()
        self.U_rsd = list()

        self.result_pref = result_pref

    def train_common(self, rank, rand_seed=1):
        np.random.seed(rand_seed)
        self.rank = rank

    def norm_U(self, mode):
        """
        Return rank-wise square norm
        """
        return np.sqrt(np.sum(self.U[mode][:self.mapper.X_dim[mode], :] ** 2, axis=0))

    def get_prodnorm_U(self):
        prodnorm_U = np.ones(self.rank)
        for l in xrange(self.mapper.order):
            prodnorm_U = prodnorm_U * self.norm_U(l)
        return prodnorm_U
    
    def reorder_U(self):
        """
        Sort columns (topic) of U.
        """
        prodnorm_U = self.get_prodnorm_U()
        topic_ind_desc = np.argsort(-prodnorm_U)
        for l in xrange(self.mapper.order):
            self.U[l] = self.U[l][:, topic_ind_desc]
        return topic_ind_desc

    def rescale_U(self):
        """
        Normalize Us to having the same norm.
        """
        prodnorm_U = self.get_prodnorm_U()
        for l in xrange(self.mapper.order):
            scale = np.power(prodnorm_U, 1.0 / self.mapper.order) / self.norm_U(l)
            self.U[l] *= scale
        return prodnorm_U

    def resize_U(self):
        for l in xrange(self.mapper.order):
            self.U[l] = self.U[l][range(self.mapper.X_dim[l]), :]

    def reorganize_U(self):
        """
        Resize, rescale and reorder Us.
        """
        self.resize_U()
        topic_order = self.reorder_U()
        self.rescale_U()
        return topic_order
    
    def write_blacklist(self, file_prefix=None):
        file_name = file_prefix + "blacklist.txt"
        fp = codecs.open(file_name, "w", encoding="utf-8")
        
        fp.write('BLACK LIST:\n')
        for l in xrange(self.mapper.order):
            for i in self.mapper.blacklist[l]:
                fp.write(self.mapper.reverse[l][i].decode('utf-8') + ' ')
            fp.write('\n')
            
        fp.write('\nSTOP COMBINATION:\n')
        for l in xrange(self.mapper.order):
            for comb in self.mapper.stopcomb[l]:
                for i in comb:
                    fp.write(self.mapper.reverse[l][i].decode('utf-8') + ' ')
                fp.write('\n')    
        fp.close()
            
    def write_topics(self, eta, N_word=10, file_prefix=None, is_tfidf=False):
        file_name = file_prefix + "topics.txt"

        ### convert scalar eta to np.array
        if np.isscalar(eta):
            eta = eta * np.ones(self.rank)
        
        modes_iterator = xrange(self.mapper.order)

        topic_score = self.get_prodnorm_U()
        topic_order = np.argsort(-topic_score)

        top_N_words = list()
        for l in xrange(self.mapper.order):
            top_N_words.append(np.zeros((self.rank, N_word), dtype=int))
            for r in xrange(self.rank):
                top_N_words[l][r, :] = np.argsort(-self.U[l][:self.mapper.X_dim[l], r])[:N_word]

        fp = codecs.open(file_name, "w", encoding="utf-8")
        for enum, r in enumerate(topic_order):
            #if topic_order[r] >= self.rank / 2:
            #    fp.write("*")
                
            fp.write("TOPIC %d Norm:%.4f eta:%.4f\n" % (r + 1, topic_score[r], eta[r]))
            for w in xrange(N_word):
                for l in modes_iterator:
                    i = top_N_words[l][r, w]
                    fmt = "%.2f %1.1f %s\t" % (self.U[l][i, r], np.log10(self.doc_freq[l][i]), self.mapper.reverse[l][i])
                    fp.write(fmt.decode('utf-8'))
                fp.write('\n')
            fp.write('\n')
        fp.close()

    def write_Us(self, file_prefix=None, topM=0, is_volume=False):
        """
        Save extracted features in (gzipped) CSV format.
        """
        #self.reorganize_U()

        for l in xrange(self.mapper.order):
            file_name = file_prefix + "U" + str(l) + ".csv.gz"
            np.savetxt(file_name, self.U[l][:self.mapper.X_dim[l], :], delimiter=",")

        if topM > 0 or is_volume:
            volume = list()
            for mode in xrange(self.mapper.order):
                volume.append(np.sum(self.U[mode], axis=0))

        if topM > 0:
            U_new = np.zeros((topM, self.rank))
            for mode in xrange(self.mapper.order):
                U_mode = self.U[mode][:self.mapper.X_dim[l], :]
                U_new = -np.sort(-U_mode, axis=0)[:topM, :] * volume[int(not mode)]
                
                #for k in xrange(self.rank):
                #    U_new[:, k] = -np.sort(-self.U[mode][:, k])[:topM] * const[k]

                np.savetxt('%sU%d_top%d.csv.gz' % (file_prefix, mode, topM), U_new)

        if is_volume:
            np.savetxt(file_prefix + 'volume.csv.gz',
                       np.sum(self.U[0], axis=0) * np.sum(self.U[1], axis=0))

    def get_error_relative(self, X=None, U=None):
        """
        Calculate ||X - U o V o W||^2 / ||X||^2
        X must be dictionary or dok_matrix.
        Only for matrix case
        """
        assert self.mapper.order == 2, \
               'Currently only support for matrix cases'
        
        if X is None:
            X = self.X
        if U is None:
            U = self.U
        
        UV = np.dot(self.U[0][:self.mapper.X_dim[0], ], \
                    self.U[1][:self.mapper.X_dim[1], ].T)
        X_arr = np.zeros(self.mapper.X_dim)
        for (key, val) in X.iteritems():
            X_arr[key] = val
            
        try:  
            return  np.linalg.norm(X_arr - UV) / np.linalg.norm(X_arr)
        except ZeroDivisionError:
            return 0


#    def get_error_RMSE(self, itr, X=None, U=None):
#        """
#        Calculate ||X - U o V o W||_NNZ.
#        X must be dictionary or dok_matrix.
#        """
#        if X is None:
#            X = self.X
#        if U is None:
#            U = self.U
#
#        denom = 1#min(itr, self.mapper.N)
#        pred = np.ones(self.rank)
#        err = 0.0
#        nnz = 0
#        #for (key, val) in X.iteritems():
#        for (key, val) in X.items():  # iteritems is not supported for pysparse (used in pyemf)
#            if np.isnan(val):
#                continue
#            pred[:] = 1   # initialization
#            for l in xrange(self.mapper.order):
#                pred *= U[l][key[l], :]
#            err += (np.double(val) / denom - np.sum(pred)) ** 2
#            #err += (val / itr) ** 2
#            #err += (np.sum(pred)) ** 2
#            nnz += 1
#        try:
#            return np.sqrt(err / nnz)
#        except ZeroDivisionError:
#            return 0
            
