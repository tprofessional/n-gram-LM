# ngram.py

import collections as clt
import math
import numpy as np
# utils.py
from utils import *


#----Load the Training Data---
def read(file):
    with open(file,'r') as f:
        sentences = f.readlines()
    return sentences

# load data in main.py instead?

#---- Tokenizing Data (count frequency of each token, set up <UNK>, <STOP> tokens) ----
'''
args:
    sentences: array of training data with length n, n = number of samples

returns dictonary of tokenized data with token counts as dict values
'''

'''need to rewrite tokenize'''
def processing(sentences):
    # count of <STOP> = numSamples
    # actually jsut convert the '\n' token to <STOP> at the end so we dont have to recount
    print(sentences)
    # returns a container that stores elements as dict keys with their counts as dict values
    token_counts = clt.Counter()
    
    # list to store all tokens from training data
    tokens = []
    
    # dictionary implementation
    tokens_counts = clt.defaultdict()

    # Loop over each sentence in input list of sentences
    for sentence in sentences:
        # print(sentence)
        #loop over each word in sentence
        for token in sentence:
            # increment count of word
            token_counts[token] += 1
            # get a list of tokens split by spaces for each sentence
            toks, token_counts = tokenize(sentence, token_counts)
            # append current sentence's tokens to parent tokens
            tokens.append(toks)

    print("token counts: ", token_counts)
    # print("tokens (in processing()): ", tokens)

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
        #iterate over each n-gram in the sentence(continuous sequence of n words)
        # n-gram is represented as a tuple of words (as far as i can understand)
        for i in range(n-1, len(sentences)): # split the n-gram into prefix and token
            ngram = tuple(sentence[i - n+1 : i + 1]) #
            prefix = ngram[:-1] # first n-1 words
            token = ngram[-1] # last word
            counts[prefix][token] += 1 # increment count of token
    return counts

#---- Calculating probability of a sentence ----
# take the output of count_ngrams()
def probabilities(counts):
    # empty dict for probabilities of tokens
    probabilities = clt.defaultdict(clt.Counter)
    # iterate over each prefix and associated Counter object
    for prefix, counter in counts.items():
        # for each prefix, it calculates total count of all tokens
        total = sum(counter.values())
        for token, count in counter.items():
            # calculate probability of token given the prefix and count of token
            # divided by total count and store value in dictionary
            probabilities[prefix][token] = count / total
        return probabilities 


def probabilities_add_one(counts, alpha=1):
    # empty dict with Counter objects as values
    probabilities = clt.defaultdict(clt.Counter)
    
    # Initialize an empty set to store the unique tokens
    vocabulary = set()

    # Loop over each Counter object in the counts dictionary
    for counter in counts.values():
        # Loop over each token in the Counter object
        for token in counter.keys():
            # Add the token to the set of unique tokens
            vocabulary.add(token)

    # Calculate the size of the vocabulary
    vocabulary_size = len(vocabulary)

    # Loop over each prefix and counter in the counts dictionary
    for prefix, counter in counts.items():
        # Calculate the total count for this prefix
        total = sum(counter.values()) + vocabulary_size * alpha
        for token, count in counter.items():
            # Calculate the probability of this token given the prefix
            probabilities[prefix][token] = (count + alpha) / total
    return probabilities



#---- Calculating perplexity of unseen corpus (dev or test data)----
# is n the number of grams
def perplexity(sentence, probabilities, n):
    # might only need <START> since <STOP> is included in vocab already
    # prepends n-1 <START> tokens and appends <STOP> token to the sentence
    sentence = ['<START>']*(n-1) + sentence + ['<STOP>']
    log_prob = 0 # variable to keep track of total log probability of sentence

    # iterate over each n-gram in the sentence
    for i in range(n-1, len(sentence)):
        ngram = tuple(sentence[i-n+1:i+1])
        # split n-gram into prefix and token
        prefix = ngram[:-1]
        token = ngram[-1]
        #retrieves the proabbility of token from the dict
        # if token is not in dictionary, it defaults to 0
        prob = probabilities[prefix].get(token, 0)
        # update log probability. if prob > 0 add neg log of prob.
        # if prob is 0, add pos infinity bc log of 0 is DNE. 
        # for perplexity, prob of 0 = infinite perplexity
        log_prob += -math.log(prob) if prob > 0 else float('inf')
    return math.exp(log_prob / (len(sentence) - n + 1))
    

if __name__ == "__main__":
    #NOTE: only if we need
    # Read and preprocess the data
    # get data as an np array of each sentence
    # shape is (61530, )
    sentences = read('./A2-Data/smaller.train.tokens')
    print(len(sentences))
    tokens = processing(sentences)
    print("length after tokenizing: ", len(tokens))

    # should return np array of all tokens from the training data
    # tokens = np.array(processing(sentences))

    # Calculate probabilities for unigram, bigram, and trigram models
    unigram_counts = count_ngrams(sentences, 1)
    bigram_counts = count_ngrams(sentences, 2)
    trigram_counts = count_ngrams(sentences, 3)

    unigram_probs = probabilities(unigram_counts)
    bigram_probs = probabilities(bigram_counts)
    trigram_probs = probabilities(trigram_counts)

    # Calculate perplexity for a test sentence
    test_sentence = np.array(read('./A2-Data/hdtv.test.tokens'))
    print("test sentence: ", test_sentence)

    print('-------- Probabilities --------')
    print(unigram_probs)
    print(bigram_probs)
    print(trigram_probs)

    print('-------- Perplexities --------')
    print(perplexity(test_sentence, unigram_probs, 1))
    print(perplexity(test_sentence, bigram_probs, 2))
    print(perplexity(test_sentence, trigram_probs, 3))
    


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