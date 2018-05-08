from __future__ import division

import json
from math import log


class MissingNgramDataException(Exception):
    def __init__(self, message):

        # Call the base class constructor with the parameters it needs
        super(MissingNgramDataException, self).__init__(message)


class LogProbs(object):

    def __init__(self, adj_value=0.001):
        self.ngram_jsons = {}
        self.adj_value = adj_value
        self.total_counts = {}

    def add_ngram_data(self, n, filename):
        self.total_counts[n] = 0
        print('Loading {}-gram data from: {}'.format(n, filename))
        with open(filename, 'r') as f:
            self.ngram_jsons[n] = json.load(f)

            # calculate the totals
            for v in self.ngram_jsons[n].values():
                c = v['count']
                self.total_counts[n] += c

    def log_prob(self, top, bot):
        return log( (top + self.adj_value) / (bot + 1) )

    def has_ngram_data(self, n):
        return n in self.ngram_jsons.keys()

    def get_log_prob(self, n, w):
        return self.get_prob_helper(n, w, 'log_prob')

    def get_prob_helper(self, n, w, p_type):
        if not self.has_ngram_data(n):
            raise MissingNgramDataException('There is no {}-gram data loaded.'.format(n))
        try:
            return self.ngram_jsons[n][w][p_type]
        except KeyError:
            return self.log_prob(0, self.total_counts[n])

    @staticmethod
    def _ngram(input_list, n):
        return zip(*[input_list[i:] for i in range(n)])

    def ngram_log_probs(self, n, sentence):
        res = []
        # add the unigram as the first token
        for gram in self._ngram(sentence, n):
            res.append(self.get_log_prob(n, ' '.join(gram)))
        return res


"""
path1 = '/data/qu009/data/ngrams/unigram_counts.json'
path2 = '/data/qu009/data/ngrams/bigram_counts.json'

lp = LogProbs()
lp.add_ngram_data(1, path1)
lp.add_ngram_data(2, path2)
"""
