import cupy as np 
import cupy as np 
import sklearn
import math
import sys
import argparse
from operator import itemgetter
import copy
import random
class DataProcessor:
	def remove_new_line(self, file, out):
		with open(file, 'r') as f, open(out, 'w+') as g:
			for line in f:
				temp =  line.replace('\n', ' ')
				g.write(temp)
		print ("Created merged dataset...\n")

	def compute_vocab(self, file):
		j = 0
		vocab={}
		text = []
		with open(file, 'r') as f:
			for line in f:
				text = line.split() #Since only 1 line exists
		for word in text:
			j+=1
			if word in vocab:
				vocab[word]+=1
				continue
				#print (word, vocab[word])
			vocab[word] = 1
		#Sort the vocabulary
		sorted_vocab = sorted(vocab.items(), key = itemgetter(1))
		#Also compute probabilities in text
		prob_vocab =  {}
		no_vocab={}
		for key, value in sorted_vocab:
			#print ("%s %s" % (key, value))
			prob_vocab[key] = (math.sqrt( 10000* float(value)/j ) + 1) * (float(1.0)/(10000* float(value)/j))
			no_vocab[key] = float(value)
		#Sort the probability vocabulary dictionary
		sorted_prob_vocab =  sorted(prob_vocab.items(), key = itemgetter(1))
		print ()
		print ("Unique: ",len(vocab))
		print ("Total words: ", j)
		return prob_vocab, no_vocab

	def make_training_data(self, file, prob_vocab,no_vocab,n=3 ):
		text = []
		with open(file ,'r') as f:
			for line in f:
				text = line.split()
		data_raw = []
		l=0
		for i in range(n+1, len(text)-n):
			if (no_vocab[text[i]] <  15):
				continue
			l+=1
			temp_context = [text[j] for j in range(i-n, i+n+1) if (j!=i and no_vocab[text[j]] >= 15)]
			temp_context.insert(0, text[i]) # Add the major word at the start of the list.
			data_raw.append(temp_context)
		print ("Length after removing: ", len(data_raw))

		#Make word to integer encoding
		int_to_words={}
		words_to_int = {}
		x=0
		for i, val in enumerate(data_raw):
			if (val[0] in words_to_int):
				continue
			#x+=1
			words_to_int[val[0]] = x
			int_to_words[x] = val[0]
			x+=1
		print ("Unique after removing: ", x)

		return words_to_int, int_to_words, data_raw


class SkipGramNegativeSampling:
    def __init__(self, vocab_size, embed_size, learning_rate, epochs, X_train, Y_train, words_to_int, int_to_words, negative_samples=5):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.lr = learning_rate
        self.epochs = epochs
        self.X_train = X_train
        self.Y_train = Y_train
        self.words_to_int = words_to_int
        self.int_to_words = int_to_words
        self.negative_samples = negative_samples

        # Initialize weight matrices
        self.w_hidden = np.random.randn(vocab_size, embed_size)
        self.w_output = np.random.randn(embed_size, vocab_size)

        self.model = {}

    def one_hot(self, word_idx):
        vec = np.zeros(self.vocab_size)
        vec[word_idx] = 1
        return vec

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def get_negative_samples(self, target_word_idx):
        neg_samples = []
        while len(neg_samples) < self.negative_samples:
            negative_sample = random.randint(0, self.vocab_size - 1)
            if negative_sample != target_word_idx:
                neg_samples.append(negative_sample)
        return neg_samples

    def build_skipgram_model(self):
        print("No. of training samples are: ", len(self.X_train))
        for epoch in range(self.epochs):
            print(f"We are at epoch: {epoch}")
            for i in range(len(self.X_train)):

                # Forward propagation for the Skip-Gram model
                input_word_idx = self.words_to_int[self.X_train[i]]
                h = np.dot(self.w_hidden.T, self.one_hot(input_word_idx))
                
                for context_word in self.Y_train[i]:
                    context_word_idx = self.words_to_int[context_word]
                    
                    # Calculate positive sample
                    output_positive = np.dot(self.w_output.T, h)
                    positive_pred = self.sigmoid(np.dot(output_positive[context_word_idx], h))

                    # Backpropagate for the positive sample (correct context word)
                    positive_error = 1 - positive_pred
                    dw_output_positive = np.outer(h, positive_error)
                    dw_hidden_positive = np.outer(self.one_hot(input_word_idx), np.dot(self.w_output, positive_error))

                    self.w_output[:, context_word_idx] += self.lr * dw_output_positive[:, context_word_idx]
                    self.w_hidden[:, input_word_idx] += self.lr * dw_hidden_positive[:, input_word_idx]

                    # Negative sampling: Get negative samples and update weights
                    negative_samples = self.get_negative_samples(context_word_idx)
                    for neg_word_idx in negative_samples:
                        output_negative = np.dot(self.w_output.T, h)
                        negative_pred = self.sigmoid(np.dot(output_negative[neg_word_idx], h))

                        # Backpropagate for the negative sample (incorrect context word)
                        negative_error = 0 - negative_pred
                        dw_output_negative = np.outer(h, negative_error)
                        dw_hidden_negative = np.outer(self.one_hot(input_word_idx), np.dot(self.w_output, negative_error))

                        self.w_output[:, neg_word_idx] += self.lr * dw_output_negative[:, neg_word_idx]
                        self.w_hidden[:, input_word_idx] += self.lr * dw_hidden_negative[:, input_word_idx]

            # Save the model weights after each epoch
            print(f"Saving model after epoch {epoch}...")
            for key, value in self.int_to_words.items():
                self.model[value] = self.w_hidden[key].reshape(1, self.w_hidden.shape[1])

            np.save(f'./utils/skipgram_negative_sampling_{epoch}.npy', self.model)

        print("Model training completed.")



# Usage
def train_with_neg_sampling(inp, out, dimensions, lr, win, epochs, neg_samples):
    processor = DataProcessor()
    prob_vocab, no_vocab = processor.compute_vocab(inp)
    words_to_int, int_to_words, data_raw = processor.make_training_data(inp, prob_vocab, no_vocab, win)

    X = []
    Y = []

    for i, val in enumerate(data_raw):
        X.append(val[0])
        Y.append(val[1:])

    model = SkipGramNegativeSampling(words_to_int, int_to_words, X, Y, lr, dimensions, epochs, neg_samples)
    model.build_skipgram_with_neg_sampling()

parser = argparse.ArgumentParser()
parser.add_argument('--train', help='Training file', dest='inp', default='./data/')
parser.add_argument('-m', '--model', help='Output model file', dest='out')
parser.add_argument('-d', '--dim', help='Dimensionality of word embeddings', dest='dimensions', default=300, type=int)
parser.add_argument('-r', '--rate', help='Learning rate', dest='lr', default=0.025, type=float)
parser.add_argument('-w', '--window', help='Max window length', dest='win', default=3, type=int)
parser.add_argument('-e', '--epochs', help='Number of training epochs', dest='epochs', default=1, type=int)
parser.add_argument('-n', '--neg_samples', help='Number of negative samples', dest='neg_samples', default=5, type=int)
args = parser.parse_args()

train_with_neg_sampling(args.inp, args.out, args.dimensions, args.lr, args.win, args.epochs, args.neg_samples)
