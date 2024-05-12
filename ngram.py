# ngram.py

import collections as clt
import math
import numpy as np
# utils.py
from utils import *
import pandas as pd # delete later


#----Load the Training Data---
# def read(file):
#     with open(file,'r') as f:
#         sentences = f.readlines()

def read_data(path):
    train_frame = pd.read_csv(path + 'dev.csv')

    # You can form your test set from train set
    # We will use our test set to evaluate your model
    try:
        test_frame = pd.read_csv(path + 'test.csv')
    except:
        test_frame = train_frame

    return train_frame, test_frame

# load data in main.py instead?

#---- Tokenizing Data (count frequency of each token, set up <UNK>, <STOP> tokens) ----
'''
args:
    sentences: array of training data n x f, n = number of samples and f = number of features

returns dictonary of tokenized data with token counts as dict values
'''
def processing(sentences):
    # returns a container that stores elements as dict keys with their counts as dict values
    token_counts = clt.Counter()
    # list to store all tokens from training data
    tokens = []
    
    # Loop over each sentence in input list of sentences
    for sentence in sentences:
        #loop over each word in sentence
        for token in sentence:
            # increment count of word
            token_counts[token] += 1
            # get a list of tokens split by spaces for each sentence
            toks, token_counts = tokenize(sentence)
            # append current sentence's tokens to parent tokens
            tokens.append(toks)

    # for words that occur less than 3 times, convert them into the UNK token
    # output_sentences = [] 
    final_tokens = [] # list to hold unique tokens including <UNK> and <STOP>
    
    for word in tokens:
        # if count of this word is less than 3
        if token_counts[word] < 3:
            # replace the word with <UNK> 
            final_tokens.append('<UNK>')
        else:
            # else append the word as it is
            final_tokens.append(word)

            # append output sentence with necesary replacements to the list of output sentences
    
            # output_sentences.append(final_tokens)
    return final_tokens


#---- Using train data to determine vocab set V ----

#---- Estimating paramters by computing bi/trigram and unigram counts ----
def count_ngrams(sentences, n):
    # empty dict with empty counter objects
    counts = clt.defaultdict(clt.Counter) 
    for sentence in sentences:
        #iterate over each n-gram (continuous sequence)
        for i in range(n-1, len(sentences)):
            ngram = tuple(sentence[i - n+1 : i + 1])
### unfinished ###
    return counts
#---- Calculating probability of a sentence ----
def probabilities(counts):
    # empty dict for probabilities of tokens
    probabilities = clt.defaultdict(clt.Counter)
    # iterate over each prefix and associated Counter object
    for prefix, counter in counts.items():
        # for each prefix, it calculates total count of all tokens
        total = sum(counter.values())
        for token, count in counter.items():
            probabilities[prefix][token] = count / total
        return probabilities 

#---- Calculating perplexity of unseen corpus (dev or test data)----
def perplexity(sentence, probabilities, n):
    sentence = ['<START>']*(n-1) + sentence + ['<STOP>']
    

if __name__ == 'main()':
    read_data('../A2-Data')
    pass




'''
class BigramFeature(FeatureExtractor):
    """Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self):
        self.vocab = set() 
        self.bigrams = {} 

    def fit(self, text_set):
        # vocabulary and count bi-grams
        for text in text_set:
            tokens = text.split()
            for i in range(len(tokens) - 1):
                bigram = (tokens[i], tokens[i+1])
                self.vocab.add(bigram)
                self.bigrams[bigram] = self.bigrams.get(bigram, 0) + 1

    def transform(self, text):
        # single text
        features = np.zeros(len(self.vocab))
        tokens = text.split()
        for i in range(len(tokens) - 1):
            bigram = (tokens[i], tokens[i+1])
            if bigram in self.vocab:
                features[list(self.vocab).index(bigram)] += 1
        return features

    def transform_list(self, text_set):
        # list of texts
        return [self.transform(text) for text in text_set]
'''