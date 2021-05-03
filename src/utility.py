import nltk
from nltk import sent_tokenize

def write_sentences(fpaths):
	"""
	Reformats a file so that each line contains only a single sentence

	:param fpaths: (list) -> a list of tuples: [ ... ,(input_fp, out_fp), ... ]

	:return: None
	"""

	for input_fp, output_fp in fpaths:
		with open(input_fp) as inp:
			sentences = sent_tokenize(inp.read())
			with open(output_fp, "w") as op:
				for sent in sentences:
					op.write(sent + "\n")

write_sentences([("../data/train/hgg_train.txt", "../data/train/hgg_train_sent.txt"), 
				("../data/train/fish_train.txt", "../data/train/fish_train_sent.txt"),
				("../data/train/restaurant_train.txt", "../data/train/restaurant_train_sent.txt"),
				("../data/train/timetravel_train.txt", "../data/train/timetravel_train_sent.txt"),
				("../data/train/worldwar_train.txt", "../data/train/worldwar_train_sent.txt"),
				("../data/train/universe_train.txt", "../data/train/universe_train_sent.txt")
				])


