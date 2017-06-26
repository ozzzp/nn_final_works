import os

import torch


class Corpus(object):
    def __init__(self, path):
        self.train = self.get_series(os.path.join(path, 'train_prediction.txt'))
        self.test = self.get_series(os.path.join(path, 'test_prediction.txt'))

    def get_series(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        # Tokenize file content
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                tokens += 1

        with open(path, 'r') as f:
            ids = torch.FloatTensor(tokens)
            token = 0
            for line in f:
                ids[token] = float(line)
                token += 1

        return ids
