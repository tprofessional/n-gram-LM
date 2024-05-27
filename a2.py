# ngram.py
import collections as clt
import math
import sys


#----Load the Training Data---
def read(file):
    with open(file,'r') as f:
        sentences = f.readlines()
    return sentences

#---- Tokenizing Data (count frequency of each token, set up <UNK>, <STOP> tokens) ----

'''Split the sentences into tokens

Arguments:
    sentence {str} -- sentence to be tokenized, such as "I love NLP"
    tk_counts {Count container} -- collections.Counter() object to keep track of the number of each token

Keyword Arguments:
    pattern {str} -- reg-expression pattern for tokenizer (default: {default_pattern})

Returns:
    tk_counts -- Counter of tokenized words, such as ['I', 'love', 'nlp'] and their frequencies
'''
def split(sentence, tk_counts):
    # split tokens by spaces
    tokens = sentence.split()
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

''' Replace all the tokens that are considered unk with the unk token
Args:
    sentences: array of training data
    all_counts: dictionary of all token counts including the unk tokens

Returns:
    text: list of lists of each line after being processed with unk, start (?), and stop tokens

'''
def tokenize(sentences, all_counts, n):
    text = []
    for sentence in sentences:
        # get the sentence as a list of its tokens
        snt = sentence.split()
        # if the word has a count less than 3, replace it with the unk token
        snt = [word if all_counts[word] >= 3 else '<UNK>' for word in snt]
        # put stop token at the end
        snt.append('<STOP>')
        #insert start tokens
        snt = insert_start(n, snt)
        # add the processed sentence to the list of all processed sentences
        text.append(snt)
    return text

'''Process the test data based on the train vocab'''
def test_tokenize(vocab, sentences, n):
    # store tokenized sentences
    text = []
    for sentence in sentences:
        # split the sentence by spaces
        snt = sentence.split()
        # parse through each token to see if it exists in the vocab
        for i, tok in enumerate(snt):
            # replace the token with unk if it's not in the train vocab
            if tok not in vocab:
                snt[i] = '<UNK>'
        #insert start tokens
        snt = insert_start(n, snt)
        # add stop token to the sentence
        snt.append('<STOP>')
        # append the list of split tokens to text
        text.append(snt)
    # print(text)
    return text

# from ngram.py
'''
    Args:
        text: tokenized sentences (including stop, start (?))
        n: number of words in ngram (uni-, bi-, or tri-)
    Returns:
        counts: dictionary of each n gram and their counts
'''
def count_ngrams(text, n):
    # empty dict with empty counter objects
    counts = clt.defaultdict(clt.Counter)
    # # perform extra processing if working on bigram or trigram (might not need this if implented in classes)
    # if n == 2:
    #     # now each sentence has one start token at the beginning
    #     text = bigram_tokenize(text)
    # elif n == 3:
    #     # now each sentence has 2 start tokens at the beginning
    #     text = trigram_tokenize(text)

    for sentence in text:
        # iterate for every ngram in the sentence
        for i in range(len(sentence) - (n-1)):
            # ngram is the current word (i+n) and the previous n-1 words (start -> i: end -> i + n)
            ngram = sentence[i: i + n]
            # get the prefix (first n-1 words in the gram)
            prefix = tuple(ngram[:-1]) # cast prefix to tuple because that's how the dict is constructed
            # the last element of the ngram
            token = ngram[-1]
            # increment the count of the current ngram
            counts[prefix][token] += 1
    return counts


# take the output of count_ngrams() to calculate probability for a sentence of tokens
def probabilities(counts, num_vocab = 0, smoothing = False, alpha = 1):
    # empty dict for probabilities of tokens
    probabilities = clt.defaultdict(clt.Counter)
    # iterate over each prefix and associated Counter object
    for prefix, counter in counts.items():
        # for each prefix, it calculates total count of all tokens
        total_occurrences = sum(counter.values())
        for token, count in counter.items():
            # calculate probability of token given the prefix and count of token
            # divided by total count and store value in dictionary
            if smoothing:
                probabilities[prefix][token] = (count + alpha) / (total_occurrences + (num_vocab * alpha))
                # print(probabilities[prefix][token])
            else:
                probabilities[prefix][token] = count / total_occurrences

    return probabilities

''' Pass in bigram/trigram processed test text (with start tokens)
'''
def sentence_probabilities(n, probabilities, text):
    # dictionary to store each sentence and their total probabilities
    sentence_probs = clt.defaultdict()
    for s_idx, sentence in enumerate(text):
        # for each token in the sentence
        sentence_log_prob = 0
        for tok_i in range(len(sentence) - (n-1)):
            # get the ngram
            ngram = sentence[tok_i : tok_i + n]
            # get the prefix
            prefix = tuple(ngram[:-1])
            # get the token
            token = ngram[-1]
            # add the log probability of this token to the running sum
            sentence_log_prob += math.log2(probabilities[prefix][token]) if probabilities[prefix][token] > 0 else 0
        # store the sentence probability of the current sentence (stored as the first token of the sentence: prob)
        sentence_probs[s_idx] = sentence_log_prob
    return sentence_probs


def perplexity(sentence_probabilities, total_words):
    total_probs = sum(sentence_probabilities.values())
    # need to take off start count from total_words
    perplexity = 2 ** (-total_probs / total_words)
    return perplexity

'''text is tokenized sentences, n is number of grams (corresponds to number of starts to insert)'''
def insert_start(n, snt):
    # insert start tokens n-1 times
    for i in range(n-1):
        # put start token at the beginning of sentence
        snt.insert(0, '<START>')
    return snt

'''
Args:
    lambdas = list of [l1, l2, l3]
    uni_probs = unigram sentence probabilities
    ...
    data
'''
def linear_interpolation(lambdas, uni_probs, bi_probs, tri_probs, text):
    # we want everything in groups of 3
    n = 3
    l1, l2, l3 = lambdas
    sentence_log_prob = 0
    sentence_probs = clt.defaultdict()
    for s_idx, sentence in enumerate(text):
        # for each token in the sentence
        sentence_log_prob = 0
        for tok_i in range(len(sentence) - (n-1)):
            # get the ngram
            ngram = sentence[tok_i : tok_i + n]
            # get the prefix
            tri_prefix = tuple(ngram[:-1])
            bi_prefix = tuple(ngram[1:-1])
            uni_prefix = tuple(ngram[2:-1])
            # get the token
            token = ngram[-1]
            # add the log probability of this token to the running sum
            sentence_log_prob += math.log2(l1 * uni_probs[uni_prefix][token] + l2 * bi_probs[bi_prefix][token] + l3 * tri_probs[tri_prefix][token])    
        # store the sentence probability of the current sentence (stored as the first token of the sentence: prob)
        sentence_probs[s_idx] = sentence_log_prob
    return sentence_probs
        


if __name__ == "__main__":
    # Read the data
    # files = ['./A2-Data/1b_benchmark.train.tokens', './A2-Data/small.train.tokens', './A2-Data/smaller.train.tokens']
    # file = files[0]
    
    file = './A2-Data/1b_benchmark.train.tokens'
    # file = './A2-Data/half.train.tokens'
    sentences = read(file)
    print("Training on {}...".format(file))

    # preprocess the data
    tok_counts, all_counts = get_counts(sentences)

    # get the vocab
    vocab = list(tok_counts.keys())
    
    # print('unique tokens without deletion: ', len(all_counts))
    # print('after all processing: ', len(tok_counts))
    uni_text = tokenize(sentences, all_counts, 1)
    bi_text = tokenize(sentences, all_counts, 2)
    tri_text = tokenize(sentences, all_counts, 3)

    # calculating probabilities for unigram, bigram, and trigram models
    unigram_counts = count_ngrams(uni_text, 1)
    bigram_counts = count_ngrams(bi_text, 2)
    trigram_counts = count_ngrams(tri_text, 3)

    unigram_probs = probabilities(unigram_counts)
    bigram_probs = probabilities(bigram_counts)
    trigram_probs = probabilities(trigram_counts)

    '''print('\n-------- Probabilities --------\n')
    print('Unigram probs:\n', unigram_probs, '\n')
    print('Bigram probs:\n', bigram_probs, '\n')
    print('Trigram probs:\n', trigram_probs, '\n')
    '''

    # ---------- TESTING ------------
    # prompt for test file
    print("\n0: HDTV", "\t2: test data\n")
    print("1: dev data\t3: train data\n")
    FILE_IDX = int(input("Choose test data (enter 0 - 3): "))
    # read test data
    test_files = ['./A2-Data/hdtv.test.tokens', './A2-Data/1b_benchmark.dev.tokens', './A2-Data/1b_benchmark.test.tokens', './A2-Data/1b_benchmark.train.tokens']
    test_sentence = read(test_files[FILE_IDX])
    print('Test file: {}'.format(test_files[FILE_IDX]))
    # preprocess the data
    test_tok_counts, test_all_counts = get_counts(test_sentence)

    # master tokenize test data
    uni_test_text = test_tokenize(vocab, test_sentence, 1)
    bi_test_text = test_tokenize(vocab, test_sentence, 2)
    tri_test_text = test_tokenize(vocab, test_sentence, 3)

    # get sentence probabilities for the test data
    uni_snt_probs = sentence_probabilities(1, unigram_probs, uni_test_text)
    bi_snt_probs = sentence_probabilities(2, bigram_probs, bi_test_text)
    tri_snt_probs = sentence_probabilities(3, trigram_probs, tri_test_text)

    '''print('\n-------- Sentence Probabilities --------\n')
    print('Unigram sentence probs:\n', uni_snt_probs, '\n')
    print('Bigram sentence probs:\n', bi_snt_probs, '\n')
    print('Trigram sentence probs:\n', tri_snt_probs, '\n')
    '''

    # Calculate perplexity for a test sentence
    # print(test_tok_counts)
    total_occs = sum(test_tok_counts.values())

    # print("test total words: ", total_occs)

    print('\n-------- Perplexities --------\n')
    print('Unigram Perplexity: ', perplexity(uni_snt_probs, total_occs))
    print('Bigram Perplexity: ', perplexity(bi_snt_probs, total_occs))
    print('Trigram Perplexity: ', perplexity(tri_snt_probs, total_occs))
    
    print('\n-------- LINEAR INTERPOLATION --------\n')
    # optimized lambdas
    lambda1 = float(input("Please enter value for lambda 1: "))
    lambda2 = float(input("Please enter value for lambda 2: "))
    lambda3 = float(input("Please enter value for lambda 3: "))
    lambdas = [lambda1, lambda2, lambda3]

    lin = linear_interpolation(lambdas, unigram_probs, bigram_probs, trigram_probs, tri_test_text)
    # print("linear interpolation: ", lin)
    lin_perplexity = perplexity(lin, total_occs)
    print('lambda 1 = ', lambdas[0], '\tlambda 2 = ', lambdas[1], '\tlambda 3 = ', lambdas[2])
    print('\nPerplexity with linear interpolation: ', lin_perplexity)

    print('\n-------- ADDITIVE SMOOTHING --------\n')
    # get total train words
    num_vocab = len(tok_counts)
    # print("total_words: ", num_vocab)

    alphas = [1, 0.05, 0.0001, 0.0000000001]

    for a in alphas:
        unigram_probs = probabilities(unigram_counts, num_vocab, smoothing = True, alpha = a)
        bigram_probs = probabilities(bigram_counts, num_vocab, smoothing = True, alpha = a)
        trigram_probs = probabilities(trigram_counts, num_vocab, smoothing = True, alpha = a)

        # get sentence probabilities for the test data
        uni_snt_probs = sentence_probabilities(1, unigram_probs, uni_test_text)
        bi_snt_probs = sentence_probabilities(2, bigram_probs, bi_test_text)
        tri_snt_probs = sentence_probabilities(3, trigram_probs, tri_test_text)

        '''print('\n-------- Smoothed Sentence Probabilities --------\n')
        print('Unigram sentence probs:\n', uni_snt_probs, '\n')
        print('Bigram sentence probs:\n', bi_snt_probs, '\n')
        print('Trigram sentence probs:\n', tri_snt_probs, '\n')
        '''

        print('\n--- Alpha = {} ---\n'.format(a))
        # print('Alpha = ', a)
        print('Unigram Perplexity: ', perplexity(uni_snt_probs, total_occs))
        print('Bigram Perplexity: ', perplexity(bi_snt_probs, total_occs))
        print('Trigram Perplexity: ', perplexity(tri_snt_probs, total_occs))
    