#!/usr/bin/env python
import sys
import numpy as np
import argparse
import cPickle as pickle

from streamingNMF import NMF

parser = argparse.ArgumentParser()
parser.add_argument('--batch', dest='batch', action='store_true', default=False,
                   help='Learn with the batch alternating least square algorithm')
parser.add_argument('-r', '--rank', type=int, dest='rank', required=True, 
                   help='The rank of the decomposed matrix')
parser.add_argument('-i', '--input', dest='input_file', default='stdin',
                   help='The edge list of the matrix to decompose')
parser.add_argument('-p', '--prefix', dest='pref', default='out.',
                   help='The prefix of output files')
parser.add_argument('-l', '--learning_rate', dest='eta', choices=['const', 'inv', 'invsqrt'], default='const',
                   help='The strategy of learning rate')
parser.add_argument('-s', '--scale', type=float, dest='scale', default=0.1,
                   help='The scale of learning rate')
parser.add_argument('--tf', dest='tf', choices=['raw', 'log', 'sqrt'], default='log',
                   help='Term frequency')
parser.add_argument('--idf', dest='idf', type=int, choices=[True, False], default=True,
                   help='Inversed document frequency')
parser.add_argument('-f', '--filter', dest='filter', default='000',
                   help='Filter for outsider/fixed-phrase')
parser.add_argument('--pred_file', dest='pred_file', default=None,
                   help='A file that precitions for missing values are written')
parser.add_argument('--delta_topic', dest='delta_topic', type=float, default=float('inf'),
                   help='Timing of making outputs')
parser.add_argument('--delta_filtering', dest='delta_bl', type=float, default=float('nan'),
                   help='Frequency to update blacklist')
parser.add_argument('--max_itr', dest='max_itr', type=int, default=1,
                   help='Maximum number of iterations (epochs)')
parser.add_argument('--init_model', dest='init_model', default=None,
                   help='File to initialize NMF model')
parser.add_argument('--save_model', dest='save_model', default=None,
                   help='File to save current NMF model')
parser.add_argument('--l1', type=float, dest='l1', default=0,
                   help='Coefficient of L1 regularization')
parser.add_argument('--l2', type=float, dest='l2', default=1e-4,
                   help='Coefficient of L2 regularization')
parser.add_argument('--start_filtering', dest='start_bl', type=int, default=0,
                   help='Start filtering after this iterations')
parser.add_argument('--sg_filtering', dest='sg_bl',
                    choices=('minlen', 'maxlr'), default='minlen',
                    help='Segmentation strategy for MGphrase')

parser.add_argument('--verbose', type=int, dest='verbose', default=0)

args = parser.parse_args()
X_normalized = False

lambdas = dict(l1=args.l1, l2=args.l2)

#args.pref = args.input_file['name'].split('/')[-1].split('.')[0]

X_opt = dict(doc_mode=0, tf=args.tf, idf=args.idf, normalization=X_normalized)
if args.eta == 'const':
    eta_opt = dict(type='tracking', coef=args.scale)
elif args.eta == 'inv':
    eta_opt = dict(type='inv', coef=args.scale)
elif args.eta == 'invsqrt':
    eta_opt = dict(type='adversarial', coef=args.scale)
    
filter_gaps = [int(c) for c in args.filter]
if np.isnan(args.delta_bl):
    args.delta_bl = args.delta_topic

if args.verbose > 0:
    print args.input_file
    print lambdas
    print X_opt
    print eta_opt
    print filter_gaps

opt = dict(file_name=args.input_file, is_skip_header=False)
trace_opt = dict(flag=0, delta_topic=args.delta_topic, delta_bl=args.delta_bl, start_bl=args.start_bl)

if args.init_model is not None: ### load cache file
    nmf = pickle.load(file(args.init_model, 'rb'))
    nmf.mapper.decode()
    is_init = False
else:
    nmf = NMF(result_pref=args.pref)
    is_init = True


nmf.train(rank=args.rank, N_loop=args.max_itr, trace_opt=trace_opt, lambdas=lambdas, X_opt=X_opt, filter_gaps=filter_gaps, eta_opt=eta_opt, pred_file=args.pred_file, is_init=is_init, is_batch=args.batch, filter_strategy=args.sg_bl)


if args.save_model is not None: ### save cache file
    nmf.mapper.encode()
    pickle.dump(nmf, file(args.save_model, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
