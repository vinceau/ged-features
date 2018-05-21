from __future__ import division

import argparse
import codecs
import json
import os

def get_ranking_data(unigram_json, dataset_tsv):
    """Dataset is in the fce error detection format. For example:
    word\ti
    word2\ti
    word3\tc
    """
    with codecs.open(unigram_json, 'r', 'utf-8') as f:
        data = json.load(f)

    d = {
        u'correct_min': None,
        u'correct_max': None,
        u'correct_count': 0,
        u'incorrect_min': None,
        u'incorrect_max': None,
        u'incorrect_count': 0,
        u'ranking_data': []
    }

    lines_seen = set()

    with codecs.open(dataset_tsv, 'r', 'utf-8') as f:
        for line in f.readlines():
            l = line.strip()
            # we don't care about repeated lines or empty lines
            if not l or l in lines_seen:
                continue
            lines_seen.add(l)

            word, label = l.split('\t')
            prob = data[word]['prob']
            assert(prob >= 0)

            if label == 'c':
                d[u'correct_count'] += 1
                if not d[u'correct_min'] or prob < d[u'correct_min']:
                    d[u'correct_min'] = prob
                if not d[u'correct_max'] or prob > d[u'correct_max']:
                    d[u'correct_max'] = prob

            elif label == 'i':
                d[u'incorrect_count'] += 1
                if not d[u'incorrect_min'] or prob < d[u'incorrect_min']:
                    d[u'incorrect_min'] = prob
                if not d[u'incorrect_max'] or prob > d[u'incorrect_max']:
                    d[u'incorrect_max'] = prob

            d[u'ranking_data'].append([word, label, prob])

    return d


def find_threshold(data, start_at=0.00005, dec_by=0.0000001):
    """We want to find the probability that maximises the division of correct and incorrect.
    We start at the probability 'start_at' and classify all words with probability
    less than that as incorrect.
    Check how many we classified correctly, and then decrement it by 'dec_by', and repeat.
    At the end we print out the best threshold.
    """
    best_prob = None
    best_loss = None
    best_is = None
    best_cs = None

    print('Hunting for good threshold. Starting at probability {} and decrementing by {}'.format(start_at, dec_by))
    while start_at > data['incorrect_min']:
        print('Trying {}'.format(start_at))
        inc = len(list(filter(lambda x: x[1] == 'i' and x[2] <= start_at, data['ranking_data'])))
        corr = len(list(filter(lambda x: x[1] == 'c' and x[2] > start_at, data['ranking_data'])))
        iscore = float(inc)*100 / data['incorrect_count']
        cscore = float(corr)*100 / data['correct_count']
        print('Classified {}/{} ({}%)'.format(inc, data['incorrect_count'], iscore))
        print('Classified {}/{} ({}%)'.format(corr, data['correct_count'], cscore))

        loss = (inc - data['incorrect_count'])**2
        loss += (corr - data['correct_count'])**2
        loss /= 2
        print('MSE: {}'.format(loss))

        if not best_loss or loss < best_loss:
            best_prob = start_at
            best_loss = loss
            best_is = iscore
            best_cs = cscore

        print('Best prob: {}% ({}, {}, {})'.format(best_prob, best_loss, best_is, best_cs))
        start_at -= dec_by

    return best_prob

def main(args):
    data = get_ranking_data(args.unigram, args.data)
    best_prob = find_threshold(data, start_at=args.start, dec_by=args.dec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Find the optimal threshold")
    parser.add_argument("data", help="the training data to calculate threshold on")
    parser.add_argument("--unigram-json", "-u", dest="unigram", help="the unigram json file for unigram prob info", required=True)
    parser.add_argument("--start-at", dest="start", type=float, default=0.00005)
    parser.add_argument("--dec-by", dest="dec", type=float, default=0.0000001)
    main(parser.parse_args())


"""
# previous best
# 1.00000000089e-06
# new best
# Best prob: 5e-07% (12309712.0, 47.8120681713, 78.8614483047)
Trying 5e-07%         
Classified 4152/8684 (47.8120681713%)       
Classified 7536/9556 (78.8614483047%)       
MSE: 12309712.0      
Best prob: 5e-07% (12309712.0, 47.8120681713, 78.8614483047)

python calc_threshold.py fce-public.train.original.tsv -u fce_unigram.json
"""
