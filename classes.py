class NgramLM():
    def __init__(self, text, smoothing = False, alpha = 1):
        self.text = text
        self.smoothing = smoothing
        self.alpha = alpha

class Unigram(NgramLM):
    def __init__(self, text):
        # dictionary containing words and probabilities
        self.probabilities = {}
        # store the text (tokenized)
        self.text = text

    # based on train data, determine word probabilities in test data
    def train(self):
        pass

    def test(self, test_data):
        pass