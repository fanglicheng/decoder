#!/usr/bin/env python3

import os

from ngram import NgramEnumerator
from common import count_lines, INF

class RefCounter(dict):
    """one ref counter for each input"""
    def __init__(self, max_n):
        dict.__init__(self)
        self.lengths = []  # ref lengths
        self.max_n = max_n

    def count(self, line):
        """take record of ngrams in a ref line, when multiple ref lines are
        fed to the counter, the max count of a particular ngram is recorded"""
        line = line.split()
        self.lengths.append(len(line))
        tmp_counter = {}
        for n in range(self.max_n):
            enum = NgramEnumerator(n+1)
            for ngram in enum.ngrams(line):
                ngram = tuple(ngram)
                if ngram in tmp_counter:
                    tmp_counter[ngram] += 1
                else:
                    tmp_counter[ngram] = 1
        for ngram, c in tmp_counter.items():
            self[ngram] = max(c, self[ngram])

    def closest_length(self, n):
        """return a length closest to n"""
        min_diff = INF
        result = 0
        for i, l in enumerate(self.lengths):
            diff = abs(l - n)
            if diff < min_diff:
                diff = min_diff
                result = i
        return result

    def __getitem__(self, ngram):
        """return 0 for unseen ngrams"""
        return self.get(ngram, 0)

class References(object):
    def __init__(self, ref_prefix, max_n):
        self.max_n = max_n  # use up to max_n gram
        # reference files
        self.files = self.get_ref_files(ref_prefix)
        # number of references per file
        self.nref = count_lines(self.files[0])
        for filename in self.files:
            n = count_lines(filename)
            assert self.nref == n, '%s has %s lines' % (filename, n)
        # counters for ngrams
        self.counters = [RefCounter(max_n) for i in range(self.nref)]

        self.load()

    def count(self, i, ngram):
        """return count of ngram in i'th sentence"""
        return self.counters[i][ngram]

    def get_counter(self, i):
        """return a dict counting ngrams in ref i"""
        return self.counters[i]

    # ----------- begin of methods class users usually do not need------------

    def load(self):
        for filename in self.files:
            f = open(filename)
            for i, line in enumerate(f):
                counter = self.counters[i]
                counter.count(line)
            f.close()

    def get_ref_files(self, ref_prefix):
        """get names of reference files"""
        dirname, basename = os.path.split(os.path.abspath(ref_prefix))
        return [os.path.join(dirname, filename)
                for filename in os.listdir(dirname)
                if filename.startswith(basename) and \
                   filename.rsplit('.', 1)[-1].isdigit()]
