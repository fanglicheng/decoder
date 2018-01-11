#!/usr/bin/env python3

import sys
from math import exp
from os import system
import os
import time

import gflags
FLAGS = gflags.FLAGS
from extractor import Extractor, LexicalWeighter
from alignment import Alignment
from rule import Rule
import hypergraph
import logger
from logprob import elog, eexp
from digamma import digamma
from common import count_lines
from abc_parser import STRAIGHT, INVERTED, START

gflags.DEFINE_integer(
    'emtrain_log_interval',
    100,
    'EM training log output interval.')

def is_glue(rule):
    return rule.lhs in [START, STRAIGHT, INVERTED]

def base_dist(rule):
    p = 0.9
    # l = len(rule.f) + len(rule.e)
    # l = len(rule.f)
    l = len([s for s in rule.f if not s.isvar])
    if l == 0:
        return 1.0
    else:
        return (1-p)**(l-1) * p

class RuleCounter(object):
    def __init__(self):
        self.count = {}
        self.prob = {}
        self.default_prob = 1e-10
        # this flag is true on the first iteration, when no normalization has
        # been done and self.prob is not generated
        self.beginning = True

    def add_count(self, rule, c):
        # all counts are accumulated on the first iteration. after that only
        # unpruned rules (those in self.prob) are counted
        #if self.beginning or rule in self.prob:
        self.count[rule] = self.count.get(rule, 0) + c

    def normalize(self):
        self.prob = {}
        s = sum(c for c in self.count.values())
        for r, c in self.count.items():
            self.prob[r] = c/s
        self.count = {}
        self.beginning = False

    def normalize_vbdp(self, alpha, threshold):
        self.prob = {}
        s = sum(c for c in self.count.values())
        s += alpha
        for r, c in self.count.items():
            n = c + alpha*base_dist(r)
            p = exp(digamma(n))/exp(digamma(s))
            if p > threshold:
                self.prob[r] = p
        self.count = {}
        self.beginning = False

    def normalize_vbd(self):
        self.prob = {}
        alpha = 1e-5
        s = sum(c for c in self.count.values())
        s += alpha*len(self.count)
        for r, c in self.count.items():
            n = c + alpha
            self.prob[r] = exp(digamma(n))/exp(digamma(s))
        self.count = {}
        self.beginning = False

    def get_prob(self, rule, length_factor=False):
        p = self.prob.get(rule)
        if p is None:
            p = self.default_prob
        if length_factor:
            p *= base_dist(rule)
        return p

class EMTrainer(object):
    def __init__(self,
                 ffilename,
                 efilename,
                 afilename,
                 outputdir,
                 alpha,
                 threshold,
                 length_factor=False,
                 lexical_weighter=None,
                 maximize_derivation=False,
                 forbid_adjacent=True,
                 require_aligned_terminal=True):
        self.ffilename = ffilename
        self.efilename = efilename
        self.afilename = afilename
        self.outputdir = outputdir
        self.alpha = alpha
        self.threshold = threshold
        self.length_factor = length_factor
        self.lexical_weighter = lexical_weighter
        self.maximize_derivation = maximize_derivation
        self.forbid_adjacent = forbid_adjacent
        self.require_aligned_terminal = require_aligned_terminal

        self.counter = RuleCounter()

        system('rm -rf %s' % outputdir)
        system('mkdir %s' % outputdir)

    def train(self, n):
        i = 0
        while i < n:
            self.em_step(i)
            i += 1
        self.write_rules()

    def em_step(self, iteration):
        ffile = open(self.ffilename)
        efile = open(self.efilename)
        afile = open(self.afilename)
        alignments = Alignment.reader_pharaoh(ffile, efile, afile)
        dirname = os.path.join(self.outputdir,
                               'iter_%s' % str(iteration+1).rjust(3, '0'))
        os.mkdir(dirname)
        if logger.level >= 1:
            logger.writeln('\niteration %s' % (iteration+1))
        likelihood = 0
        starttime = time.time()
        for i, alignment in enumerate(alignments, 1):
            if i % FLAGS.emtrain_log_interval == 0:
                logger.writeln('%s sentences at %s secs/sent' %
                               (i, (time.time()-starttime)/i))
                starttime = time.time()
            extractor = Extractor(
                maxabslen=100000,
                maxlen=10000,
                minhole=1,
                maxvars=100000,
                lexical_weighter=self.lexical_weighter,
                forbid_adjacent=self.forbid_adjacent,
                maximize_derivation=self.maximize_derivation,
                require_aligned_terminal=self.require_aligned_terminal)
            hg = extractor.extract_hypergraph(alignment)
            if hg is None:
                continue
            # compute expected counts
            self.compute_expected_counts(hg)
            likelihood += hg.root.inside
            treefilename = os.path.join(dirname,
                                        'tree_%s' % str(i).rjust(8, '0'))
            self.write_viterbi_tree(hg, treefilename)
            #for edge in hg.edges():
            #    logger.writeln('%s %s' % (self.counter.get_prob(edge.rule),
            #                              edge.rule))
        if logger.level >= 1:
            logger.writeln('likelihood: %s' % likelihood)
        if logger.level >= 1:
            logger.writeln('normalizing...')
        self.counter.normalize_vbdp(self.alpha, self.threshold)
        if logger.level >= 1:
            logger.writeln('prob table size: %s' % len(self.counter.prob))

    def compute_expected_counts(self, hg):
        hg.set_semiring(hypergraph.LOGPROB)
        hg.set_functions(lambda edge:
                         elog(self.counter.get_prob(edge.rule,
                                                    self.length_factor)),
                         None,
                         None)
        hg.inside()
        hg.outside()
        for edge in hg.edges():
            self.counter.add_count(edge.rule,
                                   eexp(edge.posterior() - hg.root.inside))

    def write_viterbi_tree(self, hg, treefilename):
        treefile = open(treefilename, 'w')
        hg.set_semiring(hypergraph.SHORTEST_PATH)
        hg.set_functions(lambda edge:
                         -elog(self.counter.get_prob(edge.rule,
                                                     self.length_factor)),
                         None,
                         None)
        treefile.write(hg.root.best_paths()[0].tree_str())
        treefile.close()

    def write_rules(self):
        rulefilename = os.path.join(self.outputdir, 'sorted_rules')
        gluerulefilename = os.path.join(self.outputdir, 'glue_rules')
        f = open(rulefilename, 'w')
        fglue = open(gluerulefilename, 'w')
        lines = []
        for r, p in self.counter.prob.items():
            if is_glue(r):
                # glue rules have no scores given by Hiero
                r.feats.append(p)
                fglue.write('%s\n' % r)
            else:
                # replace heuristic count with EM trained count. note that
                # lexical weights are also scaled here because extractor.py
                # does this to interface with scorer.py
                r.feats[0] = 1
                r.feats = [x*p for x in r.feats]
                lines.append("%s ||| %s\n" % (' '.join(str(s) for s in r.e),
                                              str(r)))
        lines.sort()
        for line in lines:
            f.write(line)
        f.close()
        fglue.close()

if __name__ == '__main__':
    import optparse
    optparser = optparse.OptionParser()
    optparser.add_option('-n',
                         '--iter',
                         type=int,
                         default=5,
                         help='number of iterations')
    optparser.add_option('-o',
                         '--output',
                         default='em_training',
                         help='output directory')
    optparser.add_option('-a',
                         '--alpha',
                         type=float,
                         default= 1e6,
                         help='hyper parameter for base distribution')
    optparser.add_option('-t',
                         '--threshold',
                         type=float,
                         default= 1e-30,
                         help='probabilities < this threshold are pruned')
    optparser.add_option('-l',
                         '--lengthfactor',
                         action='store_true',
                         default=False,
                         help='use length penalty factor')
    optparser.add_option('-w',
                         '--lexweight',
                         nargs=2,
                         help="lexical weight tables: lex.e2f lex.f2e")
    optparser.add_option('-m',
                         '--maxderivation',
                         action = 'store_true',
                         default=False,
                         help='use hypergraph with max derivation length')
    optparser.add_option('-j',
                         '--allow-adjacent',
                         dest='forbid_adjacent',
                         action = 'store_false',
                         default=True,
                         help='allow adjacent nonterminals')
    optparser.add_option('-x',
                         '--allow-nonlexical',
                         dest="require_aligned_terminal",
                         action="store_false",
                         default=True,
                         help="allow nonlexical rules")
    opts, args = optparser.parse_args()
    ffilename = args[0] 
    efilename = args[1] 
    afilename = args[2] 
    if opts.lexweight is not None:
        (fweightfile, eweightfile) = opts.lexweight
        lexical_weighter = LexicalWeighter(fweightfile, eweightfile)
    else:
        lexical_weighter = None
    trainer = EMTrainer(ffilename,
                        efilename,
                        afilename,
                        opts.output,
                        opts.alpha,
                        opts.threshold,
                        opts.lengthfactor,
                        lexical_weighter,
                        opts.maxderivation,
                        opts.forbid_adjacent,
                        opts.require_aligned_terminal)
    trainer.train(opts.iter)
