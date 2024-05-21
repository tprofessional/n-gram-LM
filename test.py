# ngram.py
import collections as clt


#----Load the Training Data---
def read(file):
    with open(file,'r') as f:
        sentences = f.readlines()
    return sentences

#---- Tokenizing Data (count frequency of each token, set up <UNK>, <STOP> tokens) ----

'''Tokenize sentence with specific pattern

Arguments:
    text {str} -- sentence to be tokenized, such as "I love NLP"
    tk_counts {Count container} -- collections.Counter() object to keep track of the number of each token

Keyword Arguments:
    pattern {str} -- reg-expression pattern for tokenizer (default: {default_pattern})

Returns:
    list -- list of tokenized words, such as ['I', 'love', 'nlp']
'''
def tokenize(text, tk_counts):
    # split tokens by spaces
    tokens = text.split()
    # add token counts from the current sentence
    tk_counts.update(tokens)

    return tk_counts


''' Use list of all tokens to determine number of unknown counts and add stop tokens
Args:
    sentences: array of training data with length n, n = number of samples

Returns:
    dictonary of tokenized data with token counts as dict values
'''
# also take ngrams as a parameter, where n = 1, 2, or 3 and determines how text is processed
# will prob need helper functions for this
def processing(sentences):
    # define unk and stop tokens
    unk = '<UNK>'
    stop = '<STOP>'
    # returns a container that stores elements as dict keys with their counts as dict values
    tok_counts = clt.Counter()

    # Loop over each sentence in input list of sentences
    for sentence in sentences:
        # get a list of tokens split by spaces for each sentence
        tok_counts = tokenize(sentence, tok_counts)
    
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
    all_toks = tok_counts.copy() # to be used in ngram-processing

    # delete tokens which occurred less than 3 times
    for word in del_toks:
        del tok_counts[word]
   
    return tok_counts, all_toks

def unigram(sentences, all_toks):
    text = []
    for sentence in sentences:
        # get the sentence as a list of its tokens
        snt = sentence.split()
        # if the word has a count less than 3, replace it with the unk token
        snt = [word if all_toks[word] >= 3 else '<UNK>' for word in snt]
        snt.insert(0, '<START>')
        snt.append('<STOP>')
        text.append(snt)

    return text


if __name__ == "__main__":
    # Read and preprocess the data
    # sentences = read('./A2-Data/1b_benchmark.train.tokens')
    files = ['./A2-Data/1b_benchmark.train.tokens', './A2-Data/small.train.tokens', './A2-Data/smaller.train.tokens']
    file = files[1]
    sentences = read(file)
    print("num samples: ", len(sentences))
    tokens, all_toks = processing(sentences)
    print('unique tokens without deletion: ', len(all_toks))
    print('after all processing: ', len(tokens))
    print(unigram(sentences, all_toks))


    if file == files[0]:
        assert len(tokens) == 26602, "Did not identify 26,602 unique tokens"