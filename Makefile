

data := docword.nips.txt.gz
voc := vocab.nips.txt

demo: $(data) $(voc)
	zcat $(data) | tail -n +4 | head -n 200000 | python parse.py $(voc) 10 | python run_nmf.py --rank 10 --delta_topic 50 --learning_rate 'invsqrt'

$(data):
	wget https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/$@

$(voc):
	wget https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/$@