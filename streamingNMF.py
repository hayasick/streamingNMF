import numpy as np
import pyecp
from collections import defaultdict
import datetime
import modelselection_BIC

def get_error_RMSE(X, U):
    rank = U[0].shape[1]
    pred = np.ones(rank)
    err = 0.0
    nnz = 0
    for (key, val) in X.items(): 
        if np.isnan(val):
            continue
        pred[:] = 1   # initialization
        for l in xrange(self.mapper.order):
            pred *= U[l][key[l], :]
        err += (np.double(val) - np.sum(pred)) ** 2
        nnz += 1
    try:
        return np.sqrt(err / nnz)
    except ZeroDivisionError:
        return 0


class NMF(pyecp.FM_base):
    """
    streaming NMF algorithm
    """
    
    def __init__(self, result_pref=''):
        """
        """
        pyecp.FM_base.__init__(self, result_pref)
        self.tt_X = list()
        self.nonzero_ind = list()
        self.doc_freq = list()
        self.X = defaultdict(int)
        self.rng = np.random.RandomState(1)

    def predict(self, inds):
        return np.dot(self.U[0][inds[0], self.topic_ind], self.U[1][inds[1], self.topic_ind])

    def update_X(self, Xm, itr):
        """
        maintain X = sum_m X_m (sparse).
        """

        #get = self.X.get
        r = (1.0 / (itr + 1))  # ratio of moving average
        for (ind, val) in Xm.iteritems():
            self.X[tuple(ind)] *= (1 - r)
            self.X[tuple(ind)] += val * r

    def compute_tt_X(self, Xm, mode):
        """
        Compute Xm(:,V,W)
        """
        other_modes = set(range(self.mapper.order))
        other_modes.remove(mode)

        # initialize nonzero flags
        self.nonzero_ind[mode][:self.mapper.X_dim[mode]] = False

        for (ind, val) in Xm.iteritems():
            if self.nonzero_ind[mode][ind[mode]] == False:
                self.tt_X[mode][ind[mode], :] = 0
                self.nonzero_ind[mode][ind[mode]] = True
            
            self.vec[:] = 1  # initialize
            for l in other_modes:
                self.vec *= self.U[l][ind[l], :]

            if np.isnan(val):  # fill missings by current estimation
                val = self.predict(ind)#np.sum(self.vec * self.U[mode][ind[mode], :])
                
            self.tt_X[mode][ind[mode], :] += val * self.vec

    def get_diag_hessian(self, mode):
        _l = int(not mode)
        GU = np.dot(self.U[_l][:self.mapper.X_dim[_l], :].T, \
                    self.U[_l][:self.mapper.X_dim[_l], :]) # maybe not efficient
        return GU + self.l2 * np.identity(self.rank)

    def get_norm_grad(self, mode):
        """
        assume self.nonzero_ind and self.tt_X is up-to-date
        """
        norm = 0
        GU = self.get_diag_hessian(mode)
        for i in np.nonzero(self.nonzero_ind[mode])[0]:
            norm += np.sum((np.dot(GU, self.U[mode][i, :]) - self.tt_X[mode][i, :]) ** 2)
        return norm / np.sum(self.nonzero_ind[mode])

    def update_U_batch(self, Xm, mode, is_nn=True):
        """
        2nd SGD for squared loss.
        """
        GU = self.get_diag_hessian(mode)
        self.compute_tt_X(Xm, mode)
        for i in np.nonzero(self.nonzero_ind[mode])[0]:
            self.U[mode][i, :] = np.linalg.solve(GU, self.tt_X[mode][i, :] - self.l1)
            if is_nn:  # projection for non-negative
                self.U[mode][i, self.U[mode][i, :] < 0] = 0


    def update_U(self, Xm, eta, mode, is_nn=True):
        """
        2nd SGD for squared loss.
        Also update Hessian?
        """
        GU = self.get_diag_hessian(mode)

        self.compute_tt_X(Xm, mode)
        self.U[mode] *= (1 - eta)
        for i in np.nonzero(self.nonzero_ind[mode])[0]:
            # self.U[mode][i, :] += eta * np.dot(self.tt_X[mode][i, :], GU_pinv)
            # self.U[mode][i, :] += eta * np.dot(self.tt_X[mode][i, :] - self.l1, GU_pinv)
            # self.U[mode][i, :] += np.linalg.lstsq(GU, eta * (self.tt_X[mode][i, :] - self.l1))[0]
            self.U[mode][i, :] += np.linalg.solve(GU, eta * (self.tt_X[mode][i, :] - self.l1))
            if is_nn:  # projection for non-negative
                self.U[mode][i, self.U[mode][i, :] < 0] = 0


    def train_common(self, rank, rand_seed, init_U, init_guess, init_Udim):
        pyecp.FM_base.train_common(self, rank, rand_seed)

        self.res = {'training error': []}
        self.vec = np.ones(rank)  # used in compute_tt_X
        self.c = np.ones(self.mapper.order)  # accumulated eta
        self.itr = -1  # initial value of #iterations


        # initialize U by uniform random variables or specified values.
        self.rand_scale = (init_guess * (self.mapper.order + 1) / self.rank) ** (1.0 / self.mapper.order) # E[sum(U^3)] = init_guess
        if init_U is None:
            for l in xrange(self.mapper.order):
                U_shape = (init_Udim, self.rank)
                self.U.append(self.randomize_U(l, shape=U_shape, rand_coef=self.rand_scale))
        else:
            assert len(init_U) == self.mapper.order, \
                   "# modes of init_U and X should be same."
            self.U = init_U

        # initialize tt_X and non-zeros flag
        for i in xrange(self.mapper.order):
            self.tt_X.append(np.zeros((init_Udim, self.rank)))
            self.nonzero_ind.append(np.zeros([init_Udim], dtype=np.int32))   # indicator of non-zero index
            self.doc_freq.append(np.zeros([init_Udim], dtype=np.int32))   # df of tf-idf

    def filter_blacklist_LR(self, mode, check_nitem=140, is_stopcomb=False, percentail=1.645, xmin_max=None, segmentation_strategy=('maxlr', 'minlen')[1]):
        """
        filtering with log-likelihood ratio test
        """
        volumes = np.sum(self.U[int(not mode)], axis=0)
        for r in xrange(self.rank):
            sind = np.argsort(-self.U[mode][:self.mapper.X_dim[mode], r])[:check_nitem]
            LR = modelselection_BIC.get_LR(self.U[mode][sind, r] * volumes[r], xmin_max=xmin_max)
            if np.max(LR) < percentail:
                continue
            
            if segmentation_strategy == 'minlen':  # maintain MGphrase to be short as possible
                end = np.where(LR > percentail)[0][0] + 1
            elif segmentation_strategy == 'maxlr':
                end = np.argmax(LR) + 1
            else:
                assert False, 'segmentation_strategy must be maxlr or minlen'

            if not is_stopcomb:  # add stop words
                self.mapper.add_blacklist(sind[:end], mode)
                self.U[mode][sind[:end], :] = 0
            else:
                if end > 1:  # MG phrase consists of more than two words
                    self.mapper.add_stopcomb(sind[:end], mode)
                    self.U[mode][sind[:end], r] = self.U[mode][sind[end + 1], r]

    def randomize_U(self, l, shape, rand_coef=None):
        if rand_coef is None:
            rand_coef = self.rand_scale * self.c[l]
        return rand_coef * self.rng.rand(shape[0], shape[1])

    def expand_buffer(self, l, is_inplace, rand_coef=None):
        """
        expand memory size of varables in which the dimension depends on X
        """
        increase_size = self.mapper.X_dim[l]
        add_mat = self.randomize_U(l, shape=(increase_size, self.rank), rand_coef=rand_coef)
        self.U[l] = np.vstack((self.U[l], add_mat))

        if is_inplace:
            self.tt_X[l].resize(self.U[l].shape)
            self.nonzero_ind[l].resize(self.U[l].shape[0])
            self.doc_freq[l].resize(self.U[l].shape[0])
        else:
            add_zeros = np.zeros((self.mapper.X_dim[l]))
            self.tt_X[l] = np.resize(self.tt_X[l], self.U[l].shape)
            self.nonzero_ind[l] = np.resize(self.nonzero_ind[l], self.U[l].shape[0])
            self.doc_freq[l] = np.concatenate((self.doc_freq[l], add_zeros))

    def write_files(self, eta):
        io_start = datetime.datetime.now()
        self.mapper.update_reverse()
        pref = self.result_pref + 'itr' + str(self.itr+1)
        self.write_topics(eta, N_word=20, file_prefix=pref)
        self.write_Us(file_prefix=pref, topM=140, is_volume=True)
        self.write_blacklist(file_prefix=pref)
        
        return datetime.datetime.now() - io_start

    def write_summary(self, cpu_time):
        if self.result_pref is not None:
            open(self.result_pref + 'summary.txt', 'w').write('%d' % cpu_time.seconds)
            self.mapper.update_reverse()
            self.write_blacklist(file_prefix=self.result_pref)

    def train(self, rank, N_loop=1, 
              X_opt=None,
              trace_opt=dict(flag=0, delta_topic=300, delta_bl=1200),
              lambdas=dict(l1=0, l2=0),
              init_guess=0.01,
              init_Udim=10000,
              init_U=None,
              rand_seed=1,
              filter_gaps=(0) * 3,
              filter_strategy='minlen',
              eta_opt=dict(type='adversarial', coef=1),
              is_init=True,
              pred_file=None,
              is_batch=False,
              epsilon=1e-3):
        """
        Start training of ECP. 
        """

        start_time = datetime.datetime.now()
        io_time = datetime.timedelta(0)

        self.l1 = lambdas['l1']
        self.l2 = lambdas['l2']

        ### record prediction for missing values if specified
        if pred_file is not None:
            pred_fp = open(pred_file, 'w')

        ### variable initialization, etc.
        self.mapper.tfidf_opt = X_opt
        if is_init:
            self.train_common(rank, rand_seed, init_U, init_guess, init_Udim)

        ### set learning rate function
        if eta_opt['type'] == 'adversarial':
            get_eta = lambda t: eta_opt['coef'] * np.sqrt(1.0 / (2 + t))
        elif eta_opt['type'] == 'inv':
            get_eta = lambda t: eta_opt['coef'] * (1.0 / (2 + t))
        elif eta_opt['type'] == 'tracking':
            get_eta = lambda t: eta_opt['coef']
        elif eta_opt['type'] == 'hybrid':
            get_eta = lambda t: np.repeat([np.sqrt(1.0 / (2 + t)), eta_opt['coef']], rank / 2)

        modes_iterator = xrange(self.mapper.order)

        ### start training
        count_loop = 0
        while(count_loop < N_loop):

            # initialize U if batch mode
            if is_batch:
                for l in modes_iterator:
                    dim = (self.mapper.X_dim[l], self.rank)
                    self.U[l][:dim[0], :] = self.randomize_U(l, dim, rand_coef=1e-5)

            # increase iteration count
            self.itr += 1

            # get index of word sets and update mappers
            Xm, nnz = self.mapper.get_Xm(self.itr)
            if nnz == -1: # reached EOF
                count_loop += 1
            if nnz == 0:  # skip empty file
                self.err.append(float('nan'))
                continue

            # set learning rate
            eta = get_eta(self.itr)

            # expand buffers
            for l in modes_iterator:
                if self.U[l].shape[0] < self.mapper.X_dim[l]:
                    self.expand_buffer(l, is_inplace=False)

            # update Us
            for l in modes_iterator:
                self.update_U(Xm, eta, l)
                self.doc_freq[l] += self.nonzero_ind[l] # update df
                self.c[l] *= (1 - eta) # update accumulated coefficient for lazy update

            # additional update for batch learning
            if is_batch:
                err = 1e-30
                while np.abs(err - get_error_RMSE(Xm, self.U)) / err > epsilon:
                    err = get_error_RMSE(Xm, self.U)
                    if err == 0:
                        break
                    for l in modes_iterator:
                        self.update_U_batch(Xm, l)


            # update X & record log
            if trace_opt['flag'] > 0:
                self.update_X(Xm, self.itr)
                if trace_opt['flag'] > 1:
                    self.res['training error'].append(self.get_error_relative())

            # write topics & Us
            if (self.itr + 1) % trace_opt['delta_topic'] == 0:
                io_time += self.write_files(eta)

            # apply blacklist
            if (self.itr + 1) % trace_opt['delta_bl'] == 0 \
                   and (self.itr + 1) >= trace_opt['start_bl']:
                #for l in xrange(self.mapper.order):
                #    self.filter_blacklist(l, gap=filter_gaps[l])
                #if self.data_type == 'twitter':   # detect stop phrase
                #    self.filter_blacklist(1, gap=filter_gaps[2], check_nitem=20, is_stopcomb=True)
                for l in xrange(self.mapper.order):
                    if l == 0:
                        xmin_max = 1   # for outsider filtering, eliminate at most one user
                    else:
                        xmin_max = None
                    
                    if filter_gaps[l] > 0:
                        self.filter_blacklist_LR(l, check_nitem=140, xmin_max=xmin_max)
                if filter_gaps[2] > 0:
                    self.filter_blacklist_LR(1, check_nitem=140, is_stopcomb=True, segmentation_strategy=filter_strategy)
                    
            # print #iterations and error
            if self.itr & (self.itr + 1) == 0:
                print "Itr %d" % (self.itr + 1)
                ###DEBUG
                #for l in xrange(self.mapper.order):
                #    #print np.sum(self.U[l][:self.mapper.X_dim[l], :]) 
                #    print np.sum(self.normalized_U(l)[:self.mapper.X_dim[l], :])
                if trace_opt['flag'] > 0:
                    print "%s: %.3f" % ("RMSE", get_error_RMSE(self.X, self.U))

            # make prediction
            if pred_file is not None and (N_loop == 1 or count_loop == N_loop):
                cpu_time = (datetime.datetime.now() - start_time) - io_time
                self.write_prediction(pred_fp, cpu_time)

            pass ### training loop end
        cpu_time = (datetime.datetime.now() - start_time) - io_time
        self.write_summary(cpu_time)

    def write_prediction(self, pred_fp, cpu_time):
        for (i, j), (_i, _j) in self.mapper.missings.iteritems():
            pred = self.predict((i, j))
            pred_fp.write('%d,%s,%s,%f\n' % (self.itr, _i, _j, pred))
        
        pred_fp.write('%f\n' % pyecp.to_seconds_float(cpu_time)) # computation time

