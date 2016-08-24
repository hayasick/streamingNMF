import numpy as np
import codecs
from collections import defaultdict, deque
import datetime
import sys

class Map_Xm(object):
    def __init__(self, tfidf_opt=None, file_name=None):
        self.order = 2  # matrix data only
        self.X_dim = np.zeros(self.order, dtype=int)
        self.tfidf_opt = tfidf_opt
        self.idf = defaultdict(dict)

        self.mapper = list()
        self.blacklist = list()
        self.stopcomb = list()
        self.reverse = list()
        for l in xrange(self.order):
            mp = defaultdict(None)
            mp.default_factory = mp.__len__
            self.mapper.append(mp)
            self.blacklist.append(set())
            self.reverse.append(dict())
            self.stopcomb.append(list())

        if file_name is None:
            self.fp = sys.stdin
        else:
            self.fp = open(file_name, 'r')

    def apply_tfidf(self, Xm):
        """
        convert Xm by using TF-IDF. Nomarization is no longer supported
        """
        assert self.order == 2, 'TF-IDF is valid for matrix data only'
        doc_mode = self.tfidf_opt['doc_mode']
        word_mode = int(not doc_mode)

        for (k, v) in Xm.iteritems():
            if self.tfidf_opt['tf'] == 'log':
                v = np.log(v) + 1
            elif self.tfidf_opt['tf'] == 'sqrt':
                v = np.sqrt(v)
                
            if self.tfidf_opt['idf']:
                v *= np.log(float(self.X_dim[doc_mode]) / len(self.idf[k[word_mode]]))
            Xm[k] = v

        
        if self.tfidf_opt['normalization']:
            norm = np.zeros(self.X_dim[doc_mode])
            for (k, v) in Xm.iteritems():
                norm[k[doc_mode]] += v**2
            for (k, v) in Xm.iteritems():
                Xm[k] /= np.sqrt(norm[k[doc_mode]])

    def encode(self):
        for l in xrange(self.order):
            mp = dict()
            for k, v in self.mapper[l].iteritems():
                mp[k] = v
            self.mapper[l] = mp

    def decode(self):
        for l in xrange(self.order):
            mp = defaultdict(None)
            mp.default_factory = mp.__len__
            for k, v in self.mapper[l].iteritems():
                mp[k] = v
            self.mapper[l] = mp


    def update_Xdim(self):
        for l in xrange(self.order):
            self.X_dim[l] = len(self.mapper[l])
            
    def add_blacklist(self, i, mode):
        self.blacklist[mode].update(i)
            
    def in_blacklist(self, ind):
        for l in xrange(self.order):
            if ind[l] is None:
                continue
            if ind[l] in self.blacklist[l]:
                return True
        return False
    
    def add_stopcomb(self, comb, mode):
        set_comb = set(comb)
        if not self.in_stopcomb(set_comb, mode):
            self.stopcomb[mode].append(set_comb)

    def in_stopcomb(self, termset, mode):
        for comb in self.stopcomb[mode]:
            if len(comb & termset) == len(comb):
                return True
        return False
        
    def get_Xm(self, m=None):
        """
        Convert words to unique IDs and update mappers.
        """

        Xm = defaultdict(int)
        while True:
            line = sys.stdin.readline()

            if line.strip() == '':
                break
            
            triplet = line.strip().split()
            assert len(triplet) == 3, 'file format is broken : ' + line
                
            ind = (self.mapper[0][triplet[0]], self.mapper[1][triplet[1]])
            Xm[ind] += float(triplet[2])
            (self.idf[ind[1]])[ind[0]] = 1

        self.update_Xdim()

        if not line:  # EOF
            nnz = -1
        else:
            nnz = len(Xm)

        if self.tfidf_opt is not None:
            self.apply_tfidf(Xm)

        # return sparse representation of Xm
        return Xm, nnz

    def update_reverse(self):
        for l in xrange(self.order):
            len_l = len(self.reverse[l])
            for k, v in self.mapper[l].iteritems():
                if v >= len_l:
                    self.reverse[l][v] = k

    def get_reverse(self):
        for l in xrange(self.order):
            self.reverse[l] = dict((v, k) for k, v in self.mapper[l].iteritems())
        return self.reverse
    
    def write_mapper(self, file_name="", mode=0, sep=" "):
        fp = codecs.open(file_name, "w", encoding="utf-8")
        for i in xrange(len(self.reverse[mode])):
            line = ("%d" + sep + "%s\n") % (i, self.reverse[mode][i])
            fp.write(line.decode('utf-8'))
        fp.close()
        



