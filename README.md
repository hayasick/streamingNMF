# streamingNMF
A streaming algorithm of non-negative matrix factorization. For more details, see our paper:

[Real-time Top-R Topic Detection on Twitter with Topic Hijack Filtering.](http://dx.doi.org/10.1145/2783258.2783402)
Kohei Hayashi, Takanori Maehara, Masashi Toyoda, Ken-ichi Kawarabayashi.
KDD 2015

## Requirements
* python 2.X
* numpy
* argparse
 
## Demo
You can try the algorithm by the following make command:
```
make demo
```
This command will do:
 * automatically download [UCI Bag of Words NIPS Data Set](https://archive.ics.uci.edu/ml/datasets/Bag+of+Words) 
 * run the algorithm to detect topics from the co-occurrence matrix
 * output topics for every 50 iterations (1 iteration = 1 document)
The detected topics will be saved as ``out.itrXXXtopics.txt``
 
