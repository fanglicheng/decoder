#!/usr/bin/env python3

import sys
import optparse

from hypergraph import Node, Edge, Hypergraph
from phrase_hypergraph import PhraseHGEdge
from rule import Rule
from percent_counter import PercentCounter
from consensus_training import cartesian
from abc_parser import START, PHRASE
from extractor import RuleDumper
import rule_filter
import gflags
FLAGS = gflags.FLAGS

class ComposedRuleExtractor(object):
    def __init__(self, treefile):
        self.hg = self.read_tree_file(treefile)

    def extract(self):
        nrule = 0
        self.replace_nonterminals()
        outputrules = []
        for node in self.hg.topo_order():
            # all nodes have a None rule as default
            node.rules = [None]
            # TODO: nodes without incoming edges are leaf nodes (None lines)
            # they have only a None rule
            if len(node.incoming) > 0:
                edge = node.incoming[0]
                # don't generate any rule from the top level rule. the top
                # level rule is either unary (when there is a parse found, or
                # super glue (then it has too many children for cartesian)
                if edge.rule.lhs == START:
                    continue
                tail_lens = [range(len(t.rules)) for t in edge.tail]
                for idx in cartesian(tail_lens):
                    subrules = [t.rules[idx[i]]
                                for i, t in enumerate(edge.tail)]
                    rule = edge.rule.compose(subrules)
                    rule.level = sum(sub.level for sub in subrules
                                     if sub is not None) + 1
                    # compute lexical weights
                    sub_e2f = 1.0
                    for sub in subrules:
                        if sub is not None:
                            sub_e2f *= sub.feats[1]
                    sub_f2e = 1.0
                    for sub in subrules:
                        if sub is not None:
                            sub_f2e *= sub.feats[2]
                    if len(edge.rule.feats) == 3:
                        # hiero rule occurrence lex weights are weighted
                        # get the original values
                        edge_e2f = edge.rule.feats[1]/edge.rule.feats[0]
                        edge_f2e = edge.rule.feats[2]/edge.rule.feats[0]
                    else:  # glue rules, no lex values
                        edge_e2f = 1.0
                        edge_f2e = 1.0
                    rule.feats = [1.0, sub_e2f*edge_e2f, sub_f2e*edge_f2e]
                    if rule_filter.filter(rule):
                        # use only the PHRASE nonterminal (no glue nonterminal)
                        node.rules.append(rule)
                        nrule += 1
            # normalize weights like hiero and interface with scorer.py
            for rule in node.rules:
                if rule is not None:
                    rule.feats = [f/(len(node.rules) - 1) for f in rule.feats]
                    outputrule = Rule()
                    outputrule.init(PHRASE, rule.f, rule.e, rule.e2f)
                    outputrule.feats = rule.feats
                    outputrules.append(outputrule)
        return outputrules

    def replace_nonterminals(self):
        """Replace glue nonterminals with PHRASE, keep only START."""
        for edge in self.hg.edges():
            rule = edge.rule
            if rule.lhs != START:
                rule.lhs = PHRASE
            rule.f = [PHRASE if sym.isvar else sym for sym in rule.f]
            rule.e = [PHRASE if sym.isvar else sym for sym in rule.e]

    def replace_nonterminals(self):
        """Replace glue nonterminals with PHRASE, keep only START."""
        for edge in self.hg.edges():
            rule = edge.rule
            if rule.lhs != START:
                rule.lhs = PHRASE
            rule.f = [PHRASE if sym.isvar else sym for sym in rule.f]
            rule.e = [PHRASE if sym.isvar else sym for sym in rule.e]

    def read_tree_file(self, treefile):
        f = open(treefile)
        current_indent = 0
        indent_level = 2
        current_edge = None
        stack = []
        for line in f:
            # TODO: why are there None node lines?
            if '|||' in line or line.strip() == 'None':
                indent = 0
                while line[indent] == ' ':
                    indent += 1
                # TODO: hack. None lines have wrong indent in
                # max derivation viterbi trees
                # if line.strip() == 'None':
                #     indent -= 2
                if indent == current_indent + indent_level:
                    current_indent = indent
                    stack.append(node)
                elif indent < current_indent:
                    npop = (current_indent - indent)//indent_level
                    current_indent = indent
                    for i in range(npop):
                        top_tmp = stack.pop()
                        # hg = Hypergraph(top_tmp)
                        # hg.topo_sort()
                        # hg.show()
                node = Node()
                if len(stack) > 0:
                    stack[-1].incoming[0].add_tail(node)
                # TODO: why are there None nodes? these nodes have incoming
                # edges. they are just a nonterminal as a leaf.
                if line.strip() != 'None':
                    rule = Rule()
                    rule.fromstr(line)
                    edge = PhraseHGEdge()
                    edge.rule = rule
                    node.add_incoming(edge)
        hg = Hypergraph(stack[0])
        hg.topo_sort()
        f.close()
        return hg

if __name__ == '__main__':
    gflags.DEFINE_string(
        'outputdir',
        'output',
        'Output dump directory.')
    try:
        argv = FLAGS(sys.argv)  # parse flags
    except gflags.FlagsError as e:
        print('%s\nUsage: %s ARGS\n%s' % (e, sys.argv[0], FLAGS))
        sys.exit(1)
    tree_files = argv
    pcounter = PercentCounter(total=len(argv), file=sys.stderr)
    rule_dumper = RuleDumper(FLAGS.outputdir, 1000000)
    for i, treefile in enumerate(tree_files):
        print(i)
        pcounter.print_percent(i)
        extractor = ComposedRuleExtractor(treefile)
        rules = extractor.extract()
        rule_dumper.add(rules)
    # dump remaining rules in dumper
    rule_dumper.dump()
