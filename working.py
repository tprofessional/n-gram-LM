import collections as clt
import math
import random
import numpy as np

# Helper functions
''' original functions
# split the words
def tokenize(text):
    return text.split()

# returns: a list of all unique tokens, including unk token after processing
#          a dictionary of each unique word and its frequencies
def build_vocab(data, threshold=3):
    word_counts = clt.Counter()
    for line in data:
        tokens = tokenize(line)
        word_counts.update(tokens)
    vocab = {word for word, count in word_counts.items() if count >= threshold}
    # print('word counts: ', len(word_counts))
    return vocab, word_counts

# process the text by replacing unk tokens with unk token
def replace_oov(tokens, vocab):
    processed_toks = [token if token in vocab else "<UNK>" for token in tokens]
    # print(len(processed_toks))
    return processed_toks

def preprocess_data(data, vocab):
    processed_data = []
    for line in data:
        # split
        tokens = tokenize(line)
        # replace unk words with unk token
        tokens = replace_oov(tokens, vocab)
        # add processed token list to the processed data
        processed_data.append(tokens)
    print(len(processed_data)) # should be num samples
    return processed_data

# Load data
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return data
'''

def read(file):
    with open(file,'r') as f:
        sentences = f.readlines()
    return sentences

'''Split the sentences into tokens

Arguments:
    text {str} -- sentence to be tokenized, such as "I love NLP"
    tk_counts {Count container} -- clt.Counter() object to keep track of the number of each token

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
        # put start token at the front (dont know if we need this)
        # snt.insert(0, '<START>')
        # put stop token at the end
        snt.append('<STOP>')
        # add the processed sentence to the list of all processed sentences
        text.append(snt)

    return text

# Language Model classes
class UnigramLM:
    def __init__(self):
        # self.vocab = vocab
        self.model = clt.defaultdict(lambda: 0)

    def train(self, data):
        
        total_count = 0
        for sentence in data:
            for token in sentence:
                self.model[token] += 1
                total_count += 1
        for token in self.model:
            self.model[token] /= total_count

    def perplexity(self, data):
        log_prob = 0
        word_count = 0
        for sentence in data:
            for token in sentence:
                # print('token prob: ', self.model[token], '\t log_prob = ', log_prob)
                if (self.model[token] > 0):
                    log_prob += math.log(self.model[token])
                else:
                    log_prob += 0
                word_count += 1
        return math.exp(-log_prob / word_count)
    

class BigramLM:
    def __init__(self):
        # self.vocab = vocab
        self.model = clt.defaultdict(lambda: clt.defaultdict(int))
        self.bigram_counts = clt.defaultdict(int)
        self.unigram_counts = clt.defaultdict(int)


    def fit(self, data):
        self.unigram_counts = clt.defaultdict(int)
        self.bigram_counts = clt.defaultdict(int)

        for sentence in data:
            for i in range(len(sentence)):
                self.unigram_counts[sentence[i]] += 1
                if i < len(sentence) - 1:
                    self.bigram_counts[(sentence[i], sentence[i+1])] += 1

    def train(self, data):
        for sentence in data:
            for i in range(len(sentence) - 1):
                self.model[sentence[i]][sentence[i+1]] += 1
        for w1 in self.model:
            total_count = sum(self.model[w1].values())
            for w2 in self.model[w1]:
                self.model[w1][w2] /= total_count
    def calculate_bigram_probability(self, word1, word2):
        # Add 1 for smoothing
        count_word1_word2 = self.bigram_counts.get((word1, word2), 0) + 1
        count_word1 = self.unigram_counts.get(word1, 0) + len(self.unigram_counts)
        if count_word1 == 0:
            return 0
        return count_word1_word2 / count_word1
    
    def perplexity(self, data):
        log_prob = 0
        total = 0
        for sentence in data:
            for i in range(len(sentence) - 1):
                prob = self.calculate_bigram_probability(sentence[i], sentence[i+1])
                log_prob += math.log(prob + 1e-10)  # Add a small constant to avoid log(0)
                total += 1
        return math.exp(-log_prob / total)
    
    def additive_smoothing(self, alpha):
        vocab_size = len(self.model)  
        for word1 in self.model:
            total_count = sum(self.model[word1].values())
            for word2 in self.model[word1]:
                self.model[word1][word2] = (self.model[word1][word2] + alpha) / (total_count + alpha * vocab_size)
        # return bigram_model

class TrigramLM:
    def __init__(self):
        # self.vocab = vocab
        self.model = clt.defaultdict(lambda: clt.defaultdict(lambda: clt.defaultdict(lambda: 0)))

    def train(self, data):
        for sentence in data:
            for i in range(len(sentence) - 2):
                self.model[sentence[i]][sentence[i+1]][sentence[i+2]] += 1
        for w1 in self.model:
            for w2 in self.model[w1]:
                total_count = sum(self.model[w1][w2].values())
                for w3 in self.model[w1][w2]:
                    self.model[w1][w2][w3] /= total_count

    def perplexity(self, data):
        log_prob = 0
        total = 0
        for sentence in data:
            for i in range(len(sentence) - 2):
                prob = self.model[sentence[i]][sentence[i+1]][sentence[i+2]]
                log_prob += math.log(prob + 1e-10)  # Add a small constant to avoid log(0)
                total += 1
        return math.exp(-log_prob / total)

    def additive_smoothing(model, alpha):
        vocab_size = len(model)  # Assuming model is a dictionary
        for w1 in model:
            for w2 in model[w1]:
                total_count = sum(model[w1][w2].values())
                for w3 in model[w1][w2]:
                    model[w1][w2][w3] = (model[w1][w2][w3] + alpha) / (total_count + alpha * vocab_size)
        return model

# Linear Interpolation
def linear_interpolation(unigram, bigram, trigram, lambdas, data):
    log_prob = 0
    word_count = 0
    lambda1, lambda2, lambda3 = lambdas
    for sentence in data:
        for i in range(2, len(sentence)):
            p1 = unigram[sentence[i]]
            p2 = bigram[sentence[i-1]][sentence[i]]
            p3 = trigram[sentence[i-2]][sentence[i-1]][sentence[i]]
            p = lambda1 * p1 + lambda2 * p2 + lambda3 * p3
            if p > 0:
                log_prob += math.log(p)
            else:
                log_prob += 0
            word_count += 1
    return math.exp(-log_prob / word_count)

'''def linear_interpolation(unigram_model, bigram_model, trigram_model, lambdas, data):
    lambda1, lambda2, lambda3 = lambdas
    perplexity = 0
    N = 0

    for sentence in data:
        sentence = ['<s>'] + sentence + ['</s>']
        for i in range(2, len(sentence)):
            unigram_prob = unigram_model[sentence[i]] / sum(unigram_model.values())
            bigram_prob = bigram_model[sentence[i-1]][sentence[i]] / sum(bigram_model[sentence[i-1]].values())
            trigram_prob = trigram_model[sentence[i-2]][sentence[i-1]][sentence[i]] / sum(trigram_model[sentence[i-2]][sentence[i-1]].values())
            
            interpolated_prob = lambda1 * unigram_prob + lambda2 * bigram_prob + lambda3 * trigram_prob
            perplexity += -np.log(interpolated_prob)
            N += 1

    perplexity = np.exp(perplexity / N)
    return perplexity
'''

# Main function
def main():
    # Load data
    train_data = read('./A2-Data/1b_benchmark.train.tokens')
    dev_data = read('./A2-Data/1b_benchmark.dev.tokens')
    test_data = read('./A2-Data/1b_benchmark.test.tokens')

    train_toks, all_train = get_counts(train_data)
    dev_toks, all_dev = get_counts(dev_data)
    test_toks, all_test = get_counts(test_data)

    train_data = tokenize(train_data, all_train)
    dev_data = tokenize(dev_data, all_dev)
    test_data = tokenize(test_data, all_test)

    # Train models
    unigram_lm = UnigramLM()
    unigram_lm.train(train_data)
    bigram_lm = BigramLM()
    bigram_lm.train(train_data)
    trigram_lm = TrigramLM()
    trigram_lm.train(train_data)

    # Evaluate models
    print("Unigram Perplexity:")
    print(f"Train: {unigram_lm.perplexity(train_data)}")
    print(f"Dev: {unigram_lm.perplexity(dev_data)}")
    print(f"Test: {unigram_lm.perplexity(test_data)}")

    print("Bigram Perplexity:")
    print(f"Train: {bigram_lm.perplexity(train_data)}")
    print(f"Dev: {bigram_lm.perplexity(dev_data)}")
    print(f"Test: {bigram_lm.perplexity(test_data)}")

    print("Trigram Perplexity:")
    print(f"Train: {trigram_lm.perplexity(train_data)}")
    print(f"Dev: {trigram_lm.perplexity(dev_data)}")
    print(f"Test: {trigram_lm.perplexity(test_data)}")

    # Additive smoothing
    alpha = 1  
    bigram_lm.additive_smoothing(alpha) 
    print("Bigram Perplexity with Additive Smoothing:")
    print(f"Train: {bigram_lm.perplexity(train_data)}")
    print(f"Dev: {bigram_lm.perplexity(dev_data)}")

    # Linear interpolation
    lambdas = (0.1, 0.3, 0.6)
    print("Linear Interpolation Perplexity:")
    print(f"Train: {linear_interpolation(unigram_lm.model, bigram_lm.model, trigram_lm.model, lambdas, train_data)}")
    print(f"Dev: {linear_interpolation(unigram_lm.model, bigram_lm.model, trigram_lm.model, lambdas, dev_data)}")
    print(f"Test: {linear_interpolation(unigram_lm.model, bigram_lm.model, trigram_lm.model, lambdas, test_data)}")

    lambda_sets = [
        (0.1, 0.3, 0.6),
        (0.2, 0.3, 0.5),
        (0.3, 0.3, 0.4),
        (0.3, 0.4, 0.3),
        (0.4, 0.3, 0.3)
    ]

    for lambdas in lambda_sets:
        print(f"Linear Interpolation with lambdas = {lambdas}")
        print("Training Set Perplexity:")
        print(f"{linear_interpolation(unigram_lm.model, bigram_lm.model, trigram_lm.model, lambdas, train_data)}")
        print("Development Set Perplexity:")
        print(f"{linear_interpolation(unigram_lm.model, bigram_lm.model, trigram_lm.model, lambdas, dev_data)}")
        print("\n")

# Report best hyperparameters on test set
best_lambdas = (0.1, 0.3, 0.6)
print(f"Best Hyperparameters (lambdas = {best_lambdas}) Perplexity on Test Set")
# print(f"Test Set Perplexity: {linear_interpolation(unigram_lm.model, bigram_lm.model, trigram_lm.model, best_lambdas, test_data)}")
    
if __name__ == "__main__":
    main()