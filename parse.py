""" Parser for UCI bag-of-words dataset

Usage: zcat docword.$(data).txt.gz | tail -n +4 | python parse.py vocab.$(data).txt $(min_freq)
- $(data): the name of data (e.g. nips)
- $(min_freq): minimum word frequency to put into a co-occurrence matrix
"""

import sys
from collections import defaultdict

min_freq = int(sys.argv[2])
mp = dict()
for i, line in enumerate(open(sys.argv[1], 'r')):
    mp[i + 1] = line.strip()


old_i = 1
word_queue = dict()
for line in sys.stdin:
    i, j, k = [int(num) for num in line.split()]

    if i != old_i:
        for w1, f1 in word_queue.iteritems():
            for w2, f2 in word_queue.iteritems():
                print w1, w2, f1 * f2
        print
        word_queue = dict()

    if k >= min_freq:
        word_queue[mp[j]] = k
    old_i = i
