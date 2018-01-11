#!/usr/bin/env python3

class NgramEnumerator:
    def __init__(self, order):
        self.order = order
        self.HOLE = '_*_'

    def elide(self, s):
        """elide the part of the string where ngrams have been enumerated"""
        m = self.order
        if m == 1:
            return (self.HOLE,)
        if len(s) >= m:
            return s[:m-1] + (self.HOLE, ) + s[1-m:]
        else:
            return s

    def ngrams(self, s):
        """enumerate ngrams in the string"""
        m = self.order
        if len(s) < m:
            return
        for i in range(len(s) - m + 1):
            ngram = s[i:i+m]
            if self.HOLE not in ngram:
                yield ngram
