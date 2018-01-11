#!/usr/bin/env python3

import sys
import re
import time

import gflags
FLAGS = gflags.FLAGS
from mymonitor import human, memory

#gflags.DEFINE_boolean(
#    'unk',
#    False,
#    'Set this to true if your LM is trained with <unk> symbol included.')

LOGZERO = -99.0

class LanguageModel(object):
    def __init__(self, str_lm_f):
        """use srilm language model from filename"""
        self.logprob_table = {}
        self.backoff_table = {}
        fin = open(str_lm_f, 'r')
        re_line = re.compile("^([\d\-\.]+)\t([^\t]+)\t?([\d\-\.]+)?\n$")
        for str_line in fin:
            m = re_line.match(str_line)
            if m:
                word = m.group(2)
                logprob = float(m.group(1))
                self.logprob_table[word] = logprob
                if m.group(3):
                    self.backoff_table[word] = float(m.group(3))
        fin.close()

    def get(self, ngram):
        # this test is expensive?
        #if FLAGS.unk:
        #    ngram = [w if w in self.logprob_table else '<unk>' for w in ngram]
        return self.logprob(ngram)

    def logprob(self, ngram):
        result = self.logprob_table.get(' '.join(ngram))
        if result is None:
            if len(ngram) == 1:
                return LOGZERO
            else:
                result = self.logprob(ngram[1:])
                if result == LOGZERO:
                    return result
                else:
                    return result + self.backoff_table.get(' '.join(ngram[:-1]), 0.0)
        else:
            return result

if __name__ == '__main__':
    gflags.DEFINE_boolean(
        'print_each_ngram',
        False,
        'Print logprob scores for each ngram.')
    gflags.DEFINE_boolean(
        'trigram_only',
        False,
        'Score only trigrams.')
    try:
        argv = FLAGS(sys.argv)  # parse flags
    except gflags.FlagsError as e:
        print('%s\nUsage: %s ARGS\n%s' % (e, sys.argv[0], FLAGS))
        sys.exit(1)

    order = 3
    start = time.time()
    lm = LanguageModel(sys.argv[1])
    print('lm loaded in %s seconds' % (time.time() - start), file=sys.stderr)
    print('mem: %s' % human(memory()), file=sys.stderr)
    total = 0
    count = 0
    totaltime = 0
    for line in sys.stdin:
        sentlogprob = 0
        words = ['<s>'] + line.split() + ['</s>']
        for i in range(2, len(words)+1):
            ngram = words[max(i-order, 0):i]
            if FLAGS.trigram_only and len(ngram) != 3:
                continue
            start = time.time()
            p = lm.get(ngram)
            totaltime += time.time() - start
            count += 1
            sentlogprob += p
            if FLAGS.print_each_ngram:
                print(p, ngram)
        total += sentlogprob
        #print(sentlogprob)
    print('%s queries in %s seconds' % (count, totaltime), file=sys.stderr)
    print(total)
