# ngram.py

import collections as clt
# utils.py
from utils import *


#----Load the Training Data---
def read(file):
    with open(file,'r') as f:
        sentences = f.readlines()
    return sentences

def tknz(text, tk_counts):
    """Tokenize sentence with specific pattern
    
    Arguments:
        text {str} -- sentence to be tokenized, such as "I love NLP"
        tk_counts {Count container} -- collections.Counter() object to keep track of the number of each token
    
    Keyword Arguments:
        pattern {str} -- reg-expression pattern for tokenizer (default: {default_pattern})
    
    Returns:
        list -- list of tokenized words, such as ['I', 'love', 'nlp']
    """
    # split tokens by spaces
    tokens = text.split()
    # add token counts from the current sentence
    tk_counts.update(tokens)

    return tk_counts
    
# load data in main.py instead?

#---- Tokenizing Data (count frequency of each token, set up <UNK>, <STOP> tokens) ----
'''
args:
    sentences: array of training data with length n, n = number of samples

returns dictonary of tokenized data with token counts as dict values
'''

# also take ngrams as a parameter, where n = 1, 2, or 3 and determines how text is processed
# will prob need helper functions for this

def processing(sentences):
    # define unk and stop tokens
    unk = '<UNK>'
    stop = '<STOP>'

    # returns a container that stores elements as dict keys with their counts as dict values
    toks_counts = clt.Counter()

    # Loop over each sentence in input list of sentences
    for sentence in sentences:
        #loop over each word in sentence
        # get a list of tokens split by spaces for each sentence
        toks_counts = tknz(sentence, toks_counts)
    
    # list to hold tokens to delete (not sure if necessary for counter, but may be necessary for vocab list)
    del_toks = []
    # counter for the number of instances of unk tokens
    unk_counts = 0
    # check if a word occurs less than 3 times, if so classify it as unk and delete it from the counter obj
    for word in toks_counts:
        if toks_counts[word] < 3:
            # update the unk counter with the number of instances of the word
            unk_counts += toks_counts[word]
            del_toks.append(word)
    
    # init unk counter to 0 and stop counter to the number of sentences in the set
    toks_counts[unk] = unk_counts
    toks_counts[stop] = len(sentences)

    # delete tokens which occurred less than 3 times
    for word in del_toks:
        del toks_counts[word]
    
    # print("toks after processing unks:\n", toks_counts)
   
    return toks_counts


if __name__ == "__main__":
    #NOTE: only if we need
    # Read and preprocess the data
    # get data as an np array of each sentence
    # shape is (61530, )
    # sentences = read('./A2-Data/1b_benchmark.train.tokens')
    sentences = read('./A2-Data/1b_benchmark.train.tokens')
    print("num samples: ", len(sentences))
    tokens = processing(sentences)
    print("number of tokens after tokenizing: ", len(tokens)) # currently getting 24102, should be 26402 so we're missing 2500 unique tokens
    # print("tokens after tokenizing:\n", tokens)