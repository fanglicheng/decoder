#!/usr/bin/env python3

from heapq import heapify, heappush, heappop
import os

import gflags
FLAGS = gflags.FLAGS
import logger
from rule import Rule, isvar, nocat
from logprob import elog
from percent_counter import PercentCounter
import decoding_flags

gflags.DEFINE_boolean(
    'merge_rules_with_same_f',
    False,
    'Share rules with the same f to save memory.')

class RuleBin(object):
    def __init__(self, K):
        self.K = K # size limit
        self.rules = []
        if FLAGS.merge_rules_with_same_f:
            self.f = None
        self.sorted = False

    def add(self, rule):
        # experiments show this has little effect
        if FLAGS.merge_rules_with_same_f:
            if self.f:
                rule.f = self.f # all rules in bin share f
            else:
                self.f = rule.f
        heappush(self.rules, (-rule.rank_cost(), rule))
        self.prune()

    def prune(self):
        while (len(self.rules) > self.K):
            heappop(self.rules)

    def __len__(self):
        return len(self.rules)

    def iter_rules(self):
        for negcost, rule in self.rules:
            yield rule

    # called after all rules are added, used in cube pruning
    def __getitem__(self, i):
        if not hasattr(self, 'sorted_rules'):
            self.sorted_rules = [rule for negcost, rule in self.rules]
            self.sorted_rules.sort()
        return self.sorted_rules[i]

class TrieNode(dict):
    def __init__(self, rule_bin_size, sym_list):
        dict.__init__(self)
        self.rules = {}  # map from lhs to RuleBin's
        self.rule_bin_size = rule_bin_size
        self.sym_list = sym_list  # list of symbols the index into this node
        self.filled = False

    def add(self, rule):
        bin = self.rules.setdefault(rule.lhs, RuleBin(self.rule_bin_size))
        bin.add(rule)
        self.filled = True

    def sym_str(self):
        return ' '.join(str(sym) for sym in self.sym_list)

    def tree_str(self, indent=0):
        indent_str = ' '*indent
        result = '%s<TrieNode: %s>\n' % (indent_str, self.sym_str())
        for lhs, bin in self.iter_rulebins():
            result += '%slhs: %s\n' % (indent_str, lhs)
            for rule in bin:
                result += '%s%s\n' % (indent_str, rule)
        for node in self.values():
            result += node.tree_str(indent+4)
        return result

    def iter_rules(self):
        for bin in self.rules.values():
            for rule in bin.iter_rules():
                yield rule

    def iter_rulebins(self):
        for lhs, bin in self.rules.items():
            yield lhs, bin

    def get_sorted_rules(self):
        if not hasattr(self, 'sorted_rules'):
            self.sorted_rules = [rule for rule in self.iter_rules()]
            self.sorted_rules.sort()
        return self.sorted_rules
        
class Grammar(object):
    def __init__(self, rule_bin_size):
        self.rule_bin_size = rule_bin_size
        self.reset()

    def reset(self):
        self.root = TrieNode(self.rule_bin_size, ())
        self.added = 0

    def add(self, rule):
        node = self.root
        for word in rule.f:
            if FLAGS.nt_mismatch:
                if isvar(word):
                    word = nocat(word)
            node = node.setdefault(word,
                                   TrieNode(self.rule_bin_size,
                                            node.sym_list + (word,)))
        node.add(rule)
        self.added += 1

    def iter_rules(self, sym_list):
        """Iterate over rules given rhs"""
        trie_node = self.get_trie_node(sym_list)
        if trie_node is not None:
            for rule in trie_node.iter_rules():
                yield rule

    def get_sorted_rules(self, sym_list):
        """Return a sorted list of rules given rhs"""
        trie_node = self.get_trie_node(sym_list)
        if trie_node is not None:
            return trie_node.get_sorted_rules()
        else:
            return []

    def get_trie_node(self, sym_list):
        """sym_list is a list of rhs symbols. Use this list of symbols
        to index into the grammar Trie, starting from the top, and return
        the trie node found. Return None if no trie node found"""
        trie_node = self.root
        for sym in sym_list:
            trie_node = trie_node.get(sym)
            if trie_node is None:
                return None
        return trie_node

    def iter_trie_nodes(self):
        """Breadth-first iteration of trie nodes"""
        queue = [self.root]
        while queue:
            node = queue.pop()
            for child in node.values():
                queue.append(child)
            yield node

    def size(self):
        """Number of rules."""
        result = 0
        for node in self.iter_trie_nodes():
            for lhs, bin in node.iter_rulebins():
                result += len(bin)
        return result

class LexicalITG(object):
    def __init__(self, filename, rule_bin_size, features):
        """features is a Feature object (list of features)"""
        self.features = features
        self.lexgrammar = Grammar(rule_bin_size)
        self.itg = Grammar(rule_bin_size)
        self.filepath = filename
        self.filename = os.path.basename(self.filepath)
        self.name = self.filename.rsplit('.', 1)[0]
        self.nbadrules = 0

    def initialize(self):
        """grammar needs to be initialized before using"""
        self.load(self.filepath)

    def update(self, i):
        "Dummy method for compatibility."
        pass

    def load(self, filename):
        if logger.level >= 1:
            logger.writeln('loading rules from %s...' % filename)
        percent_counter = PercentCounter(input=filename, file=logger.file)
        f = open(filename)
        for i, line in enumerate(f):
            if logger.level >= 1:
                percent_counter.print_percent(i)
            try:
                rule = Rule()
                rule.fromstr(line)
            except AssertionError:
                logger.write('bad rule: %s %s: %s\n' % (filename, i, line))
                self.nbadrules += 1
                continue
            rule.grammar = self  # used in computing features scores
            self.features.score_rule(rule)
            if rule.arity == 0:
                self.lexgrammar.add(rule)
            else:
                self.itg.add(rule)
        f.close()
        if logger.level >= 1:
            logger.writeln()
            logger.writeln(self.stats())

    def stats(self):
        result = '--Rule Stats--\n'
        result += 'itg rules added: %s\n' % self.itg.added
        result += 'lexical rules added: %s\n' % self.lexgrammar.added
        result += 'itg rules pruned: %s\n' % (self.itg.added - self.itg.size())
        result += 'lexical rules pruned: %s\n' % (self.lexgrammar.added -
                                                  self.lexgrammar.size())
        result += 'bad rules: %s\n' % self.nbadrules
        return result

class SCFG(Grammar):
    def __init__(self, filename, rule_bin_size, features):
        """features is a Feature object (list of features)"""
        Grammar.__init__(self, rule_bin_size)
        self.features = features
        self.filepath = filename
        self.filename = os.path.basename(self.filepath)
        self.name = self.filename.rsplit('.', 1)[0]
        self.nbadrules = 0

    def initialize(self):
        """grammar needs to be initialized before using"""
        if not self.persentence():
            self.load(self.filepath)

    def update(self, i):
        "Update per-sentence grammar for sentence i"
        if self.persentence():
            self.reset()
            self.load(os.path.join(self.filepath,
                                   'g' + str(i).rjust(6, '0')))

    def load(self, filename):
        if logger.level >= 1:
            logger.writeln('loading rules from %s...' % filename)
        percent_counter = PercentCounter(input=filename, file=logger.file)
        f = open(filename)
        badrules = open('badrules', 'w')
        for i, line in enumerate(f):
            if logger.level >= 1:
                percent_counter.print_percent(i)
            try:
                rule = Rule()
                rule.fromstr(line)
            except AssertionError:
                badrules.write('%s %s: %s\n' % (filename, i, line))
                self.nbadrules += 1
                continue
            rule.grammar = self  # used in computing features scores
            self.features.score_rule(rule)
            self.add(rule)
        f.close()
        badrules.close()
        if logger.level >= 1:
            logger.writeln()
            logger.writeln(self.stats())

    def stats(self):
        result = '--Rule Stats--\n'
        result += 'rules: %s\n' % self.added
        result += 'rules pruned: %s\n' % (self.added - self.size())
        result += 'bad rules: %s\n' % self.nbadrules
        return result

    def persentence(self):
        "whether this is a per-sentence grammar."
        return os.path.isdir(self.filepath)


