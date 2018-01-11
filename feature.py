# feature classes have a 'transition' method, which, given a rule and its
# antecedents, computes a feature cost and possibly a new state.

from ngram import NgramEnumerator
from logprob import elog
import logger

import gflags
FLAGS = gflags.FLAGS
from mymonitor import human, memory

gflags.DEFINE_boolean(
    'use_python_lm',
    False,
    'Use the python lm module in place of the swig srilm wrapper.')

class Feature(object):
    def __init__(self): 
        """features should do only trivial jobs in __init__. more serious
        jobs should be postponed to 'initialize' method"""
        # a feature is stateless if it can be computed from the rule alone
        self.stateless = True
        
        # if a feature is not stateless, this index is set by add_feature so
        # that feature knows the index of its state in an item's state list
        self.i = None

        # feature name is used for printing information, weight matching, etc
        self.name = self.__class__.__name__

    def __repr__(self):
        return self.name

    def weight(self, deduction):
        """the funtion that computes the feature value for a hyperedge"""
        return 0, None # None means stateless

    def heuristic(self, item):
        """Computes the heuristic value to be added to an item."""
        return 0

class MockLM(Feature):
    def __init__(self, m, lmfile):
        Feature.__init__(self)
        self.stateless = False
        self.ngram_enum = NgramEnumerator(m)

    def weight(self, deduction):
        vars = [item.state[self.i] for item in deduction.tail]
        s = tuple(deduction.rule.rewrite(vars))
        return self.ngram_cost(s), self.ngram_enum.elide(s)

    def ngram_cost(self, s):
        cost = 0
        for ngram in self.ngram_enum.ngrams(s):
            cost += 0
        return -cost #LM returns neg logprob

class LM(Feature):
    def __init__(self, m, lmfile):
        Feature.__init__(self)
        self.stateless = False
        self.m = m
        self.lmfile = lmfile
        self.ngram_enum = NgramEnumerator(self.m)

        if FLAGS.use_python_lm:
            from python_lm import LanguageModel
        else:
            from swig_lm import LanguageModel

        logger.writeln('reading LM: %s' % self.lmfile)
        if FLAGS.use_python_lm:
            self.lm = LanguageModel(self.lmfile)
            self.getcost = self.lm.get
        else:
            self.lm = LanguageModel(self.m, self.lmfile)
            self.getcost = self.lm

    def weight(self, deduction):
        vars = [item.state[self.i] for item in deduction.tail]
        s = tuple(deduction.rule.rewrite(vars))
        return self.ngram_cost(s), self.ngram_enum.elide(s)

    def ngram_cost(self, s):
        cost = 0
        for ngram in self.ngram_enum.ngrams(s):
            cost += self.getcost(ngram)
        return -cost #LM returns neg logprob

    def heuristic(self, item):
        s = item.state[self.i]
        if item.i == 0:
            prefix = ('<s>',) * (self.m - 1)
        else:
            prefix = ('<unk>',) * (self.m - 1)
        if item.rightmost:
            suffix = ('</s>',) * (self.m - 1)
        else:
            suffix = ()
        s = prefix + s + suffix
        h = 0
        for ngram in self.ngram_enum.ngrams(s):
            h += self.lm(ngram)
        return -h

class NTMismatch(Feature):
    def __init__(self):
        Feature.__init__(self)
        self.stateless = False

    def weight(self, deduction):
        return sum(nt != item.var
                   for nt, item in zip(deduction.rule.fnts(),
                                       deduction.tail)), None

class LengthPenalty(Feature):
    def __init__(self):
        Feature.__init__(self)

    def weight(self, rule):
        return len(rule.e) - rule.arity

class ITG(Feature):
    def __init__(self):
        Feature.__init__(self)

    def weight(self, rule):
        return -elog(rule.feats[0])

def is_phrase(rule):
    return rule.grammar is not None and 'phrase' in rule.grammar.name

def is_glue(rule):
    return rule.grammar is not None and 'glue' in rule.grammar.name

def is_ghkm(rule):
    return rule.grammar is not None and 'ghkm' in rule.grammar.name

def is_hiero(rule):
    return rule.grammar is not None and 'hiero' in rule.grammar.name

# phrase features

class PhraseFE(Feature):
    def weight(self, rule):
        if is_phrase(rule):
            return -elog(rule.feats[0])
        else:
            return 0

class PhraseLexFE(Feature):
    def weight(self, rule):
        if is_phrase(rule):
            return -elog(rule.feats[1])
        else:
            return 0

class PhraseEF(Feature):
    def weight(self, rule):
        if is_phrase(rule):
            return -elog(rule.feats[2])
        else:
            return 0

class PhraseLexEF(Feature):
    def weight(self, rule):
        if is_phrase(rule):
            return -elog(rule.feats[3])
        else:
            return 0

class PhraseCount(Feature):
    def weight(self, rule):
        if is_phrase(rule):
            return 1
        else:
            return 0

# deprecated, for backward compatibility with older config files
class GHKM(Feature):
    def weight(self, rule):
        if is_ghkm(rule):
            return -elog(rule.feats[0])
        else:
            return 0

# ghkm features

class GHKMGlobal(Feature):
    def weight(self, rule):
        if is_ghkm(rule):
            return -elog(rule.feats[0])
        else:
            return 0

class GHKMLHSConditional(Feature):
    def weight(self, rule):
        if is_ghkm(rule):
            return -elog(rule.feats[1])
        else:
            return 0

class GHKMRHSConditionalEC(Feature):
    def weight(self, rule):
        if is_ghkm(rule):
            return -elog(rule.feats[2])
        else:
            return 0

class GHKMRHSConditionalCE(Feature):
    def weight(self, rule):
        if is_ghkm(rule):
            return -elog(rule.feats[3])
        else:
            return 0

class GHKMLexicalEC(Feature):
    def weight(self, rule):
        if is_ghkm(rule):
            return -elog(rule.feats[4])
        else:
            return 0

class GHKMLexicalCE(Feature):
    def weight(self, rule):
        if is_ghkm(rule):
            return -elog(rule.feats[5])
        else:
            return 0

class GHKMCount(Feature):
    def weight(self, rule):
        if is_ghkm(rule):
            return 1
        else:
            return 0

# hiero features

class HieroLHSConditional(Feature):
    def weight(self, rule):
        if is_hiero(rule):
            return rule.feats[0]
        else:
            return 0

class HieroRHSConditionalEC(Feature):
    def weight(self, rule):
        if is_hiero(rule):
            return rule.feats[1]
        else:
            return 0

class HieroRHSConditionalCE(Feature):
    def weight(self, rule):
        if is_hiero(rule):
            return rule.feats[2]
        else:
            return 0

class HieroLexicalEC(Feature):
    def weight(self, rule):
        if is_hiero(rule):
            return rule.feats[3]
        else:
            return 0

class HieroLexicalCE(Feature):
    def weight(self, rule):
        if is_hiero(rule):
            return rule.feats[4]
        else:
            return 0

class HieroCount(Feature):
    def weight(self, rule):
        if is_hiero(rule):
            return 1
        else:
            return 0

# glue features

class GlueRule(Feature):
    def weight(self, rule):
        if is_glue(rule):
            return -elog(rule.feats[0])
        else:
            return 0

class GlueRuleCount(Feature):
    def weight(self, rule):
        if is_glue(rule):
            return 1
        else:
            return 0
	
class UnknownWord(Feature):
    def weight(self, rule):
        if hasattr(rule, 'is_unknown'):
            return 1
        else:
            return 0
