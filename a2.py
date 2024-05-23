# ngram.py
import collections as clt
import math
import numpy as np
import sys


#----Load the Training Data---
def read(file):
    with open(file,'r') as f:
        sentences = f.readlines()
    return sentences

#---- Tokenizing Data (count frequency of each token, set up <UNK>, <STOP> tokens) ----

'''Split the sentences into tokens

Arguments:
    text {str} -- sentence to be tokenized, such as "I love NLP"
    tk_counts {Count container} -- collections.Counter() object to keep track of the number of each token

Keyword Arguments:
    pattern {str} -- reg-expression pattern for tokenizer (default: {default_pattern})

Returns:
    tk_counts -- Counter of tokenized words, such as ['I', 'love', 'nlp'] and their frequencies
'''
def split(text, tk_counts):
    # split tokens by spaces
    tokens = text.split()
    # add token counts from the current sentence
    tk_counts.update(tokens)

    return tk_counts

''' Use list of all tokens to determine number of unknown counts and add stop tokens
Args:
    sentences: array of training data with length n, n = number of samples

Returns:
    tok_counts - dictionary of tokenized data with token counts as dict values & omits unk tokens
    all_counts - dictonary of all tokenized data with token counts as dict values
'''
def get_counts(sentences):
    # define unk and stop tokens
    unk = '<UNK>'
    stop = '<STOP>'
    # returns a container that stores elements as dict keys with their counts as dict values
    tok_counts = clt.Counter()

    # Loop over each sentence in input list of sentences
    for sentence in sentences:
        # get a list of tokens split by spaces for each sentence
        tok_counts = split(sentence, tok_counts)
    
    # list to hold tokens to delete (not sure if necessary for counter, but may be necessary for vocab list)
    del_toks = []
    # counter for the number of instances of unk tokens
    unk_counts = 0
    # check if a word occurs less than 3 times, if so classify it as unk and delete it from the counter obj
    for word in tok_counts:
        if tok_counts[word] < 3:
            # update the unk counter with the number of instances of the word
            unk_counts += tok_counts[word]
            del_toks.append(word)
    
    # init unk counter to 0 and stop counter to the number of sentences in the set
    tok_counts[unk] = unk_counts
    tok_counts[stop] = len(sentences)

    # all unique tokens without deleting unk token counts
    all_counts = tok_counts.copy() # to be used in ngram-processing
    # delete tokens which occurred less than 3 times
    for word in del_toks:
        del tok_counts[word]
   
    return tok_counts, all_counts

''' Replace all the tokens htat are considered unk with the unk token
Args:
    sentences: array of training data
    all_counts: dictionary of all token counts including the unk tokens

Returns:
    text: list of lists of each line after being processed with unk, start (?), and stop tokens

'''
def tokenize(sentences, all_counts):
    text = []
    for sentence in sentences:
        # get the sentence as a list of its tokens
        snt = sentence.split()
        # if the word has a count less than 3, replace it with the unk token
        snt = [word if all_counts[word] >= 3 else '<UNK>' for word in snt]
        # put stop token at the end
        snt.append('<STOP>')
        # add the processed sentence to the list of all processed sentences
        text.append(snt)

    return text

''' my prob funct
def probabilities(tk_counts):
    # empty dict for probabilities of tokens
    probabilities = clt.defaultdict(clt.Counter())
    # iterate over each prefix and associated Counter object
    for word in tk_counts:
        # for each prefix, it calculates total count of all tokens
        total_occurrences = sum(tk_counts.values())
        for token, count in tk_counts.items():
            # calculate probability of token given the prefix and count of token
            # divided by total count and store value in dictionary
            probabilities[token] = count / total_occurrences
        return probabilities
'''
    
# from ngram.py
'''
    Args:
        text: tokenized sentences (including stop, start (?))
        n: number of words in ngram (uni-, bi-, or tri-)
    Returns:
        counts: dictionary of each n gram and their counts
'''

def count_ngrams(text, n):
    print('n = ', n)
    # empty dict with empty counter objects
    counts = clt.defaultdict(clt.Counter) 
    for sentence in text:
        #iterate over each n-gram in the sentence(continuous sequence of n words)
        # iterate num_words - (n-1) times
        for i in range(len(sentence) - (n-1)): # split the n-gram into prefix and token
            # ngram is the current word and the previous n-1 words (start -> i - n + 1 : end -> i + 1)
            ngram = list(sentence[i: i + n]) # n-gram is represented as a list of words
            prefix = tuple(ngram[:-1]) # first n-1 words
            token = ngram[-1] # the last element of the ngram
            counts[prefix][token] += 1
    
    return counts
            

# take the output of count_ngrams() to calculate probability for a sentence
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


def perplexity(sentence, trigram_probs, n):
    if not isinstance(sentence, list):
        sentence = sentence.tolist()
    sentence = ['<START>'] + sentence + ['<STOP>']
    log_prob = 0

    for i in range(n-1, len(sentence)):
        ngram = tuple(sentence[i-n+1:i+1])
        prefix = ngram[:-1]
        token = ngram[-1]

        # Check if trigram_probs is not None and prefix exists in trigram_probs
        if trigram_probs is not None and prefix in trigram_probs:
            prob = trigram_probs[prefix].get(token, 0)
        else:
            prob = 0

        log_prob += -math.log(prob) if prob > 0 else 0

    return math.exp(log_prob / (len(sentence) - n + 1))


if __name__ == "__main__":
    # Read the data
    files = ['./A2-Data/1b_benchmark.train.tokens', './A2-Data/small.train.tokens', './A2-Data/smaller.train.tokens']
    file = files[1]
    sentences = read(file)
    # print("num samples: ", len(sentences))

    # preprocess the data
    tok_counts, all_counts = get_counts(sentences)
    # print('unique tokens without deletion: ', len(all_counts))
    # print('after all processing: ', len(tok_counts))
    text = tokenize(sentences, all_counts)
    # print(text)

    # calculating probabilities for unigram, bigram, and trigram models
    unigram_counts = count_ngrams(text, 1)
    bigram_counts = count_ngrams(text, 2)
    trigram_counts = count_ngrams(text, 3)

    unigram_probs = probabilities(unigram_counts)
    bigram_probs = probabilities(bigram_counts)
    trigram_probs = probabilities(trigram_counts)

    # Calculate perplexity for a test sentence
    test_sentence = read('./A2-Data/hdtv.test.tokens')
    print("test sentence: ", test_sentence)
    
    print('\n-------- Probabilities --------\n')
    print('Unigram probs:\n', unigram_probs, '\n')
    print('Bigram probs:\n', bigram_probs, '\n')
    print('Trigram probs:\n', trigram_probs, '\n')

    print('\n-------- Perplexities --------\n')
    print('Unigram Perplexity: ', perplexity(test_sentence, unigram_probs, 1))
    print('Bigram Perplexity: ', perplexity(test_sentence, bigram_probs, 2))
    print('Trigram Perplexity: ', perplexity(test_sentence, trigram_probs, 3))


    if file == files[0]:
        assert len(tok_counts) == 26602, "Did not identify 26,602 unique tokens"