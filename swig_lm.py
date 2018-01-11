#!/usr/bin/env python3

import srilm, sys

class LanguageModel(object):
    """wrapper of language model, note that methods ending with 'Prob'
    returns a negative log probability (base 10).
    this is a class wrapper around the c swig-srilm wrapper, thus only one 
    instance of this class is allowed"""

    def __init__(self, m, filename):
        """use srilm language model from filename"""
        self.m = m #order
        self.model = srilm.initLM(self.m)
        success = srilm.readLM(self.model, filename)
        if not success:
            print('LM load error')
            sys.exit(1)

    def getUnigramProb(self, s):
        return srilm.getUnigramProb(self.model, ' '.join(s))

    def getBigramProb(self, s):
        return srilm.getBigramProb(self.model, ' '.join(s))

    def getTrigramProb(self, s):
        return srilm.getTrigramProb(self.model, ' '.join(s))

    def getSentenceProb(self, s):
        """this has a limit of 15 words in max, don't use it,
        instead, use the function below to evaluate a sentence"""
        return srilm.getSentenceProb(self.model, ' '.join(s), len(s))

    def getSentenceCost(self, s):
        """return a positive cost of a sentence, <s> and </s> are
        added and accounted for, used by MER training"""
        cost = 0
        s = ('<s>',)*(self.m-1) + s + ('</s>',)
        for i in range(len(s) - 2):
            cost += self.getTrigramProb(s[i:i+3])
        return -cost

    def getSentenceProb_ns(self, s):
        """same with 'getSentenceProb' except special marks <s> and </s>
        are not taken into account, used in heuristcs"""
        prob = 0
        if len(s) >= 1:
            prob += self.getUnigramProb(s[0:1])
        if len(s) >= 2:
            prob += self.getBigramProb(s[0:2])
        if len(s) >= 3:
            for i in range(3, len(s)+1):
                prob += self.getTrigramProb(s[i-3:i])
        return prob
        

    def getCorpusProb(self, filename):
        return srilm.getCorpusProb(self.model, filename)

    def getCorpusPpl(self, filename):
        return srilm.getCorpusPpl(self.model, filename)


    def trans_heuristic(self, s):
        """give a heuristic score to the beginning m-1 words of a partial
        translation"""
        cost = 0
        if len(s) >= 1:
            cost += self.getUnigramProb(s[0:1])
        if len(s) >= 2:
            cost += self.getBigramProb(s[0:2])
        return -cost

    def rule_heuristic(self, ruleitem):
        cost = 0
        for seg in ruleitem.r.t:
            cost += self.getSentenceProb_ns(seg)
        return -cost

    def __call__(self, s):
        """although this language model only supports up to 3-gram. we expect
        the common interface to be: lm([ngram]) -> prob"""
        if self.m == 1:
            return self.getUnigramProb(s)
        elif self.m == 2:
            return self.getBigramProb(s)
        elif self.m == 3:
            return self.getTrigramProb(s)
        else:
            raise ValueError()

