"""
Usage:
path1 = '/data/qu009/data/ngrams/unigram_counts.json'
path2 = '/data/qu009/data/ngrams/bigram_counts.json'
wf = WordFeatures(path1, path2)
wf.preprocess(all_sents)

where all_sents is a list of all sentences in the training data

wf.features(sentence)
returns a 3-tuple
1. individual word level features as torch.sparse.FloatTensor
2. unigram features
3. bigram features
"""


from __future__ import division


from sparse import ZeroOneRowSparse, SimpleRowSparse
from log_prob import LogProbs

import string
import torch

import argparse
import logging
import numpy as np
from nltk import pos_tag
import json
import re


class NoWordFeaturesErrorException(Exception):
    def __init__(self):
        Exception.__init__(self, 'No word features were enabled')


"""
Word level features:

* begins with capital letter or not (good for captilisation errors)
* ends with s (plural related errors)
* starts with a vowel (an, a)
* contains numbers
* contains symbols
* is only symbol (no alphanum)
"""

class WordSparse(object):

    def __init__(self, num_chars=2):
        """num_chars is the number of letters of the beginning and end to encode
        """
        self.start_char2id = {
            'UNKCP': 0
        }
        self.end_char2id = {
            'UNKCP': 0
        }
        self.num_chars = num_chars
        self.static_feats = [
            self.has_doubles,
            self.is_capital,
            self.trailing_s,
            self.starting_vowel,
            self.contains_num,
            self.contains_sym,
            self.only_sym,
        ]

    def preprocess(self, word):
        self.start_char_index(word, addunk=True)
        self.end_char_index(word, addunk=True)

    def start_chars(self, w):
        start_cp = w[:self.num_chars]
        while len(start_cp) < self.num_chars:
            start_cp += ' '
        return start_cp

    def end_chars(self, w):
        end_cp = w[-self.num_chars:]
        while len(end_cp) < self.num_chars:
            end_cp = ' ' + end_cp
        return end_cp

    def start_char_index(self, word, addunk=False):
        start = self.start_chars(word)
        if start not in self.start_char2id:
            if not addunk:
                return self.start_char2id['UNKCP']
            self.start_char2id[start] = len(self.start_char2id)
        return self.start_char2id[start]

    def end_char_index(self, word, addunk=False):
        end = self.end_chars(word)
        if end not in self.end_char2id:
            if not addunk:
                return self.end_char2id['UNKCP']
            self.end_char2id[end] = len(self.end_char2id)
        return self.end_char2id[end]

    @staticmethod
    def has_doubles(word):
        if bool(re.search(r'([a-zA-Z])\1', word)):
            return 1
        return 0

    @staticmethod
    def is_capital(word):
        if word[0].isupper():
            return 1
        return 0

    @staticmethod
    def trailing_s(word):
        if word[-1].lower() == 's':
            return 1
        return 0

    @staticmethod
    def starting_vowel(word):
        if word[0].lower() in 'aeiou':
            return 1
        return 0

    @staticmethod
    def contains_num(word):
        for l in word:
            if l.isdigit():
                return 1
        return 0

    @staticmethod
    def contains_sym(word):
        for l in word:
            if l in string.punctuation:
                return 1
        return 0

    @staticmethod
    def only_sym(word):
        for l in word:
            if l not in string.punctuation:
                return 0
        return 1

    def sparse_length(self):
        return len(self.static_feats) + len(self.start_char2id) + len(self.end_char2id)

    def word2feat(self, word):
        feats = [ f(word) for f in self.static_feats ]
        feat_sparse = SimpleRowSparse(feats)
        start_idx = self.start_char_index(word)
        start_sparse = ZeroOneRowSparse.from_index(start_idx, len(self.start_char2id))
        end_idx = self.end_char_index(word)
        end_sparse = ZeroOneRowSparse.from_index(end_idx, len(self.end_char2id))

        joined = feat_sparse + start_sparse + end_sparse
        # for peace of mind, make sure it's the same size
        assert(joined.size[1] == self.sparse_length())

        return joined

    def sent2feat(self, sentence):
        return [self.word2feat(w) for w in sentence]


class POSSparse(object):

    def __init__(self, window=1, min_pos_freq=1, join_posgrams=True):
        # pad ends with zero
        self.pos2id = {
            'PAD': 0,
            'UNK': 1,
        }
        self.posgram2id = {
            'UNKPOS': 0
        }
        self.min_pos_freq = min_pos_freq
        self.posgram_counts = {}
        self.join_posgrams = join_posgrams
        self.window = window

    def sent2pos(self, sentence, addunk=False):
        """Converts a sentence to POS id
        """
        ret = []
        pos = pos_tag(sentence)
        for _, p in pos:
            if p not in self.pos2id:
                if not addunk:
                    ret.append(self.pos2id['UNK'])
                    continue
                else:
                    self.pos2id[p] = len(self.pos2id)
            ret.append(self.pos2id[p])
        return ret

    def sparse_length(self):
        if self.join_posgrams:
            return len(self.posgram2id)
        else:
            return (1 + self.window * 2) * len(self.pos2id)

    def sent2posgram(self, sentence, addunk=False):
        """Converts a sentence into POSgram id
        """
        ret = []
        ids = self.sent2pos(sentence, addunk)
        for i in range(len(sentence)):
            block = []
            for j in range(i - self.window, i + 1 + self.window):
                if j < 0 or j >= len(sentence):
                    block.append(0)
                else:
                    block.append(ids[j])

            if not self.join_posgrams:
                # we are going to keep the posgram in a list form
                ret.append(block)
                continue

            # we are using posgrams so assign each posgram a unique id
            pg = str(block)

            if addunk:  # we're still processing so add unknowns to dictionary
                if pg not in self.posgram2id:
                    self.posgram2id[pg] = len(self.posgram2id)
                    self.posgram_counts[pg] = 0
                self.posgram_counts[pg] += 1

            else:  # we're not processing anymore, return UNKPOS if it's not in dict
                if pg not in self.posgram2id or self.posgram_counts[pg] < self.min_pos_freq:
                    pg = 'UNKPOS'

            ret.append([self.posgram2id[pg]])

        return ret


    def sent2pos_sparse(self, sentence):
        pg = self.sent2posgram(sentence)
        ret = []
        for p in pg:
            if self.join_posgrams:
                ret.append(ZeroOneRowSparse(p, len(self.posgram2id)))
            else:
                sparses = [ZeroOneRowSparse([pos_id], len(self.pos2id)) for pos_id in p]
                joined = reduce((lambda x, y: x + y), sparses)
                ret.append(joined)
        return ret


class WordFeatures(object):

    def __init__(self, unigrams=None, bigrams=None, min_pos_freq=1, join_posgrams=True, use_static=True, use_posgrams=True, use_word_positions=True):
        self.ps = POSSparse(window=1, min_pos_freq=min_pos_freq, join_posgrams=join_posgrams)
        self.ws = WordSparse(num_chars=2)

        # load the ngram data
        self.log_probs = LogProbs()
        if unigrams:
            self.log_probs.add_ngram_data(1, unigrams)
        if bigrams:
            self.log_probs.add_ngram_data(2, bigrams)

        # make sure at least one of the features is enabled
        if not any([use_static, use_posgrams, use_word_positions]):
            raise NoWordFeaturesErrorException

        self.use_static = use_static
        self.use_posgrams = use_posgrams
        self.use_word_positions = use_word_positions

    def preprocess(self, all_sentences):
        for s in all_sentences:
            # compute sentence posgram combinations
            self.ps.sent2posgram(s, addunk=True)
            for w in s:
                # compute all the start and end character pair combinations
                self.ws.preprocess(w)

    def sparse_length(self):
        total = 0
        if self.use_static:
            total += self.ws.sparse_length()
        if self.use_posgrams:
            total += self.ps.sparse_length()
        if self.use_word_positions:
            # plus two for word pos from start, and from end
            total += 2
        return total

    def word_position_sparse(self, sent):
        """Return a sparse vector with:
        word position in relation to start,
        word position in relation to end
        """
        res = []
        for i, w in enumerate(sent):
            res.append(SimpleRowSparse([i + 1, len(sent) - i]))
        return res

    def sent2feat(self, sentence):
        # first get sentence only word features
        # list of Sparse() for each word
        features = []
        if self.use_static:
            word_sparse = self.ws.sent2feat(sentence)
            features.append(word_sparse)

        # for each word in the sentence, a sparse for each pos trigram
        if self.use_posgrams:
            pos_sparse = self.ps.sent2pos_sparse(sentence)
            features.append(pos_sparse)

        # for each word in the sentence, the position of the word in relation from the start, and from the end
        if self.use_word_positions:
            positions = self.word_position_sparse(sentence)
            features.append(positions)

        # make sure we have a feature for each word
        assert(all(len(sentence) == len(f) for f in features))

        # join them together
        sparses = []
        for res in zip(*features):
            joined = reduce((lambda x, y: x + y), res)
            # for a peace of mind, check the size is right
            assert(joined.size[1] == self.sparse_length())
            sparses.append(joined)

        return sparses

    def sent2feat_stacked(self, sentence):
        """
        The word features will be stacked together
        as a single tensor of size: sentence_length x self.sparse_length()

        * first two characters as the prefix of the current word
        * last two characters as the suffix of the current word
        * POS trigrams centered at the current word;
        * log probability of the word unigram at position i
        * a feature called unigram bias with value 1;
        * log probability of word bi-grams starts
        * each one comes with a bias feature???
        """
        # returns a list of sparses
        sparses = self.sent2feat(sentence)

        # stack them together
        res = sparses[0]
        for s in sparses[1:]:
            res = res.stack(s)

        return res

    def features(self, sentence):
        # get the sentence features
        # sentence_length x self.sparse_length()
        res = self.sent2feat_stacked(sentence)

        # unigrams
        unilp = None
        if self.log_probs.has_ngram_data(1):
            unilp = self.log_probs.ngram_log_probs(1, sentence)
        # bigrams
        bilp = None
        if self.log_probs.has_ngram_data(2):
            bilp = self.log_probs.ngram_log_probs(2, sentence)

        return res, unilp, bilp

    def features_dense(self, sentence, stacked=False):
        return self.features(sentence, stacked=stacked, dense=True)

    def batch_features(self, sents, max_sent_length, dense=False):
        batch = []
        for s in sents:
            if not s:  # we have an empty padding sentence
                vec = torch.sparse.FloatTensor(max_sent_length, self.sparse_length())
            else:
                vec = self.sent2feat_stacked(s)
                # set the number of rows
                vec.size[0] = max_sent_length
                vec = vec.to_torch()
            if dense:
                vec = vec.to_dense()
            batch.append(vec)

        return torch.stack(batch, dim=1)


"""
import json
with open('test.json', 'r') as f:
    sents = json.load(f)['data']

inp = sents[0]['input_sentence']
out = sents[0]['corrected_sentence']
ps = POSSparse()
pg = ps.sent2posgram(inp)
ws = WordSparse()
#for w in inp:
#    print(w)
#    print(ws.word2feat(w))
#print(pg)
path1 = '/data/qu009/data/ngrams/unigram_counts.json'
path2 = '/data/qu009/data/ngrams/bigram_counts.json'
wf = WordFeatures(path1, path2)

embed_size = 300
batch_size = 64
from functions import fill_batch
import generators as gens
train_txt = "fce-data/train.kaneko.txt"
gen1 = gens.word_list(train_txt)
gen2 = gens.batch(gens.sorted_parallel(gen1, embed_size*batch_size), batch_size)
batchs = [b for b in gen2]
bl = list(range(len(batchs)))

# preprocess the word features
all_sents = []
for n, j in enumerate(bl):
    batch = fill_batch([b[-1].split() for b in batchs[j]])
    all_sents.extend(batch)

wf.preprocess(all_sents)
res = wf.features('so they said to do that'.split())
"""
