import numpy as np
import collections


def tokenize(text, tk_counts):
    """Tokenize sentence with specific pattern
    
    Arguments:
        text {str} -- sentence to be tokenized, such as "I love NLP"
        tk_counts {Count container} -- collections.Counter() object to keep track of the number of each token
    
    Keyword Arguments:
        pattern {str} -- reg-expression pattern for tokenizer (default: {default_pattern})
    
    Returns:
        list -- list of tokenized words, such as ['I', 'love', 'nlp']
    """
    # token_counts = collections.Counter()
    # not sure if we need to lowercase the words
    text = text.lower()
    print(text)
    # split tokens by spaces
    tokens = text.split()
    print(tokens)
    # add token counts from the current sentence
    tk_counts.update(tokens)

    return tokens, tk_counts


class FeatureExtractor(object):
    """Base class for feature extraction.
    """
    def __init__(self):
        pass
    def fit(self, text_set):
        pass
    def transform(self, text):
        pass  
    def transform_list(self, text_set):
        pass



class UnigramFeature(FeatureExtractor):
    """Example code for unigram feature extraction
    """
    def __init__(self):
        self.unigram = {}
        
    def fit(self, text_set: list):
        """Fit a feature extractor based on given data 
        
        Arguments:
            text_set {list} -- list of tokenized sentences and words are lowercased, such as [["I", "love", "nlp"], ["I", "like", "python"]]
        """
        index = 0
        for i in range(0, len(text_set)):
            for j in range(0, len(text_set[i])):
                if text_set[i][j].lower() not in self.unigram:
                    self.unigram[text_set[i][j].lower()] = index
                    index += 1
                else:
                    continue
                    
    def transform(self, text: list):
        """Transform a given sentence into vectors based on the extractor you got from self.fit()
        
        Arguments:
            text {list} -- a tokenized sentence (list of words), such as ["I", "love", "nlp"]
        
        Returns:
            array -- an unigram feature array, such as array([1,1,1,0,0,0])
        """
        feature = np.zeros(len(self.unigram))
        for i in range(0, len(text)):
            if text[i].lower() in self.unigram:
                feature[self.unigram[text[i].lower()]] += 1
        
        return feature
    
    def transform_list(self, text_set: list):
        """Transform a list of tokenized sentences into vectors based on the extractor you got from self.fit()
        
        Arguments:
            text_set {list} --a list of tokenized sentences, such as [["I", "love", "nlp"], ["I", "like", "python"]]
        
        Returns:
            array -- unigram feature arraies, such as array([[1,1,1,0,0], [1,0,0,1,1]])
        """
        features = []
        for i in range(0, len(text_set)):
            features.append(self.transform(text_set[i]))
        
        return np.array(features)
            

class BigramFeature(FeatureExtractor):
    """Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self):
        # Add your code here!
        raise Exception("Must be implemented")
    def fit(self, text_set):
        # Add your code here!
        raise Exception("Must be implemented")
    def transform(self, text):
        # Add your code here!
        raise Exception("Must be implemented")
    def transform_list(self, text_set):
        # Add your code here!
        raise Exception("Must be implemented")

class CustomFeature(FeatureExtractor):
    """customized feature extractor, such as TF-IDF
    """
    def __init__(self):
        # Add your code here!
        raise Exception("Must be implemented")
    def fit(self, text_set):
        # Add your code here!
        raise Exception("Must be implemented")
    def transform(self, text):
        # Add your code here!
        raise Exception("Must be implemented")
    def transform_list(self, text_set):
        # Add your code here!
        raise Exception("Must be implemented")


        
