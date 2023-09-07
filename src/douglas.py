import re
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# Hyperparameters
torch.manual_seed(42)
EMBEDDING_DIM = 64
HIDDEN_DIM = 64
# LAYERS = ?

def load_checkpoint(checkpoint_fpath, model, optimizer):
	"""
	Loads checkpoint.

	:param checkpoint_fpath: (str) -> the filepath to the checkpoint
	:param model: (torch.LSTM) -> the model
	:param optimizer: (torch.optimizer) -> the optimizer

	:return: (torch.LSTM) -> model
			 (torch.optimizer) -> the optimizer
			 (int) -> the current epoch
	"""
	
	checkpoint = torch.load(checkpoint_fpath)
	model.load_state_dict(checkpoint["model_state_dict"], strict=False)
	optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
	return model, optimizer, checkpoint["epoch"]

def load_data(fpaths):
	"""
	Loads data.

	:param fpaths: (list) -> filepaths

	:return: (str) -> the corpus
 	"""

	corpus = ""
	for fp in fpaths:
		with open(fp) as inp:
			corpus += inp.read()

	return corpus

def preprocess(corpus):
	"""
	Basic preprocessing operations:
	- extraneous symbol removal
	- lowercasing
	- word tokenizing
	"""

	corpus = re.sub(r'<\w+ \/?>', ' ', corpus) # html tags
	corpus = re.sub(r'[!"$%&\\/()*,.:;<=>?@\[\]^_{}|~`\\]', ' ', corpus) #punct
	corpus = re.sub(r'\d+',' ', corpus) # numbers
	corpus = re.sub(r'\\xa0', ' ', corpus) # xa0
	corpus = re.sub(r'\n', ' ', corpus) # newlines
	corpus = re.sub(r'-(?=[a-z])', ' ', corpus) # hyphenated words
	corpus = re.sub(r'-', ' ', corpus) # random dashes
	# expand contractions
	corpus = re.sub(r"n\'t", " not", corpus)
	corpus = re.sub(r"\'re", " are", corpus)
	corpus = re.sub(r"\'s", " is", corpus)
	corpus = re.sub(r"\'d", " would", corpus)
	corpus = re.sub(r"\'ll", " will", corpus)
	corpus = re.sub(r"\'t", " not", corpus)
	corpus = re.sub(r"\'ve", " have", corpus)
	corpus = re.sub(r"\'m", " am", corpus)
	corpus = re.sub(r'\s+', ' ', corpus) # excess spaces
	corpus = corpus.lower().strip() # lowercase/strip

	return word_tokenize(corpus)

def get_encoding_maps(vocab):
	"""
	Encode vocabulary
	"""

	word_to_ix = {}
	ix_to_word = {}

	for word in vocab:
		if word not in word_to_ix:
			word_to_ix[word] = len(word_to_ix)
			ix_to_word[word_to_ix[word]] = word

	return word_to_ix, ix_to_word

def build_ngrams(corpus, n):
	"""
	Builds an ngram set of the form [([context....], target), ([context....], target)....]
	where n > 1
	"""

	return [([corpus[i], corpus[i + (n - 2)]], corpus[i + (n - 1)]) for i in range(len(corpus) - (n - 1))]

class LSTMGenerator(nn.Module):

	def __init__(self, embedding_dim, hidden_dim, vocab_size, n_layers=1):
		super(LSTMGenerator, self).__init__()
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.vocab_size = vocab_size
		self.n_layers = n_layers

		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

		self.lstm = nn.LSTM(embedding_dim, hidden_dim)

		self.hidden2word = nn.Linear(hidden_dim, vocab_size)

	def forward(self, context):
		embeds = self.word_embeddings(context)
		lstm_out, _ = self.lstm(embeds.view(len(context), 1, -1))
		# project to output space
		word_space = self.hidden2word(lstm_out.view(len(context), -1))
		word_space = torch.reshape(word_space[-1], (1, self.vocab_size))
		word_scores = F.log_softmax(word_space, dim=1)

		return word_scores

def prepare_sequence(seq, word_to_ix):
	idxs = [word_to_ix[w] for w in seq]
	return torch.tensor(idxs, dtype=torch.long)

def train(model, ngrams, n_epochs, word_to_ix):
	model.train()
	model = model.to(device)
	loss_function = nn.NLLLoss()
	optimizer = optim.SGD(model.parameters(), lr=0.1)
	avg_loss = 0
	i = 0
	for epoch in range(n_epochs):
		print(f"Starting epoch {epoch}...")
		for context, target in ngrams:
			model.zero_grad()

			context_in = prepare_sequence(context, word_to_ix).to(device)
			target = prepare_sequence(target.split(), word_to_ix).to(device)

			candidate_scores = model(context_in)

			loss = loss_function(candidate_scores, target)
			loss.backward()
			optimizer.step()
			avg_loss += loss.data.item()
			i += 1
		checkpoint = {
			"epoch": epoch + 1,
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict()
		}
		torch.save(checkpoint, "/home/output/models/unilstm.pt")
		avg_loss /= i
		print("Loss: " + str(avg_loss))
		i = 0
		avg_loss = 0

def write_story(model, primer, predict_len, temperature, word_to_ix, title):
	model.eval()
	model = model.to(device)
	with torch.no_grad():
		for i in range(predict_len):
			inp = prepare_sequence(primer.split(), word_to_ix)[-2:].to(device) #last two words as input
			word_scores = model(inp)

			# sample from network as multinomial distribution
			word_scores_dist = word_scores.data.view(-1).div(temperature).exp()
			top_i = torch.multinomial(word_scores_dist, 1)[0]

			# Add predicted word to corpus and use as next input
			predicted_word = list(word_to_ix.keys())[list(word_to_ix.values()).index(top_i)]
			primer += " " + predicted_word

	# write story to file
	with open("/home/output/stories/" + title + ".txt", 'w') as op:
		op.write(primer)

	return primer

# loading and preprocessing
train_fpaths = ["/home/data/train/hgg_train.txt", "/home/data/train/fish_train.txt", "/home/data/train/restaurant_train.txt",  "/home/data/train/timetravel_train.txt", "/home/data/train/worldwar_train.txt", "/home/data/train/universe_train.txt"]
data = load_data(train_fpaths)
corpus = preprocess(data)
# encode vocabulary
vocab = set(corpus)
word_to_ix, ix_to_word = get_encoding_maps(vocab)

trigrams = build_ngrams(corpus, n=3)
model = LSTMGenerator(EMBEDDING_DIM, HIDDEN_DIM, len(vocab))
optimizer = optim.SGD(model.parameters(), lr=0.1)
model, _, _= load_checkpoint("/home/output/models/unilstm.pt", model, optimizer)
train(model, trigrams, 100, word_to_ix)
story = write_story(model, "the history of ", predict_len=1000, temperature=0.8, word_to_ix=word_to_ix, title="hgg_spinoff")
print(story)



