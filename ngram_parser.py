from __future__ import division

import argparse
import codecs
import sys
from math import log
import json

ADJUSTMENT = 0.001

def tsv2dict(filename):
    d = {}
    with codecs.open(filename, 'r', encoding='utf8', errors='ignore') as f:
        for line in f.readlines():
            words, count = line.strip().split('\t')
            d[words] = int(count)
    return d


def adj_prob(top, bottom):
    return log((top + ADJUSTMENT) / (bottom + 1))


def main(args):

    all_counts = {}
    total_count = 0

    for fn in args.files:
        with open(fn, 'r') as f:
            for line in f.readlines():
                words, count = line.strip().split('\t')

                all_counts[words] = {
                    'count': int(count)
                }
                total_count += int(count)


    print(total_count)

    if args.unigram_file:
        print('using unigram file {}'.format(args.unigram_file))
        with open(args.unigram_file, 'r') as f:
            uni_data = json.load(f)

    for k, v in all_counts.items():
        c = v['count']

        bottom = total_count

        if args.unigram_file:
            first_word = k.split()[0]
            try:
                first_count = uni_data[first_word]['count']
                if first_count > 0:
                    print('found count of {} is {}'.format(first_word, first_count))
                    bottom = first_count
            except KeyError:
                pass

        assert bottom != 0
        assert c <= bottom

        prob = c / bottom
        log_prob = adj_prob(c, bottom)
        assert(log_prob < 1)

        v['prob'] = prob
        v['log_prob'] = log_prob

    with open('output.json', 'w') as f:
        json.dump(all_counts, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*")
    parser.add_argument("--unigram-file", required=False)
    main(parser.parse_args())
