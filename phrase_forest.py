#!/usr/bin/env python3

import sys
import time
import pickle
import os

import alignment
import common
from common import timed, select
from rule import Rule
from extractor import extract_phrases
import hypergraph
import logger
from phrase_hypergraph import PhraseHGNode, PhraseHGEdge
from alignment import get_reversed_index_map
from consensus_training import cartesian
import rule_filter
from rule_dumper import RuleDumper
import rule_extraction_flags

import gflags
FLAGS = gflags.FLAGS

gflags.DEFINE_string(
    'phrase_nonterminal',
    'A',
    'Nonterminal used for phrase forest.')
gflags.DEFINE_bool(
    'delete_unaligned',
    False,
    'Delete unaligned words in phrase decomposition forest.')

PHRASE_NT = '[%s]' % FLAGS.phrase_nonterminal

def make_rule(parent,
              children,
              fwords,
              ewords,
              parent_nt=None,
              children_nts=None):
    """Given parent and children as phrases (boxes), return a rule.
    A phrase is a list [fi, fj, ei, ej]."""
    if parent_nt is None:
        parent_nt = PHRASE_NT
    if children_nts is None:
        children_nts = [PHRASE_NT]*len(children)
    fi, fj, ei, ej = parent
    f = fwords[fi:fj]
    e = ewords[ei:ej]
    # maps from index in phrase to index in sentence
    # used to keep track of word alignment for lexical weighting
    fpos = [i for i in range(fi, fj)]
    epos = [i for i in range(ei, ej)]
    # None is used as a placeholder in gaps
    for var_idx in range(len(children)):
        child_fi, child_fj, child_ei, child_ej = children[var_idx]
        child_nt = children_nts[var_idx]
        # phrase index
        sub_fi = child_fi - fi
        sub_fj = child_fj - fi
        f[sub_fi] = child_nt
        fpos[sub_fi] = (child_fi, child_fj)
        for i in range(sub_fi+1, sub_fj):
            f[i] = None
            fpos[i] = None
        # phrase index
        sub_ei = child_ei - ei
        sub_ej = child_ej - ei
        e[sub_ei] = (child_nt, var_idx)
        epos[sub_ei] = (child_ei, child_ej)
        for i in range(sub_ei+1, sub_ej):
            e[i] = None
            epos[i] = None
    # remove placeholders
    f = [w for w in f if w is not None]
    fpos = [i for i in fpos if i is not None]
    epos = [i for i in epos if i is not None]
    # recover nonterminal permutation
    new_e = []
    e2f = []
    for w in e:
        if w is not None:
            if type(w) is tuple:
                new_e.append(w[0])
                e2f.append(w[1])
            else:
                new_e.append(w)
    # build rule
    rule = Rule()
    rule.init(parent_nt, f, new_e, e2f)
    rule.fpos = fpos
    rule.epos = epos
    return rule

def phrase_decomposition_forest(align):
    # save the index mapping so that we can restore indices after phrase
    # decomposition forest generation
    if not FLAGS.delete_unaligned:
        fmap = get_reversed_index_map(align.faligned)
        emap = get_reversed_index_map(align.ealigned)
    a = align.remove_unaligned()

    phrases = list(extract_phrases(a))
    #print('%s phrases' % len(phrases))
    n = len(a.fwords)
    #print('%s words' % n)
    chart = [[None for j in range(n+1)] for i in range(n+1)]
    for i1, j1, i2, j2 in phrases:
        #print('(%s,%s)' % (i1, i2))
        chart[i1][i2] = PhraseHGNode(PHRASE_NT, i1, i2, j1, j2)
    for s in range(1, n+1):
        for i in range(0, n-s+1):
            j = i + s
            #print('span (%s %s)' % (i, j))
            node = chart[i][j]
            if node is None:
                continue
            splits = 0
            # test for binary ambiguity
            for k in range(i+1, j):
                if chart[i][k] is not None and chart[k][j] is not None:
                    edge = PhraseHGEdge()
                    edge.add_tail(chart[i][k])
                    edge.add_tail(chart[k][j])
                    node.add_incoming(edge)
                    #print('split at %s' % k)
                    splits += 1
            # find the maximal cover if no ambiguity found
            if splits == 0:
                edge = PhraseHGEdge()
                l = i
                while l < j:
                    next = l + 1
                    m = j - 1 if l == i else j
                    while m > l:
                        if chart[l][m] is not None:
                            edge.add_tail(chart[l][m])
                            next = m
                            break
                        m -= 1
                    l = next
                node.add_incoming(edge)
    hg = hypergraph.Hypergraph(chart[0][n])
    hg.assert_done('topo_sort')
    assert len(phrases) == len(hg.nodes), \
            '%s phrases, %s nodes' % (len(phrases), len(hg.nodes))
    #if len(phrases) != len(hg.nodes):
    #    print('%s phrases, %s nodes' % (len(phrases), len(hg.nodes)))
    #for node in hg.nodes:
    #    i1,j2,i2,j2 = node.phrase
    #    print('(%s,%s)' % (i1, i2))

    # restore indices on each node
    if FLAGS.delete_unaligned:
        return hg, a
    else:
        for node in hg.nodes:
            node.fi = max(fmap[node.fi])
            node.fj = min(fmap[node.fj])
            node.ei = max(emap[node.ei])
            node.ej = min(emap[node.ej])
        return hg, align

def make_composed_rules(node, align):
    node.children = []
    node.rules = []
    if node.fj - node.fi > 10 or node.ej - node.ei > 10:
        return
    L = []
    for edge in node.incoming:
        for children in cartesian([[[tailnode]] + tailnode.children
                                  for tailnode in edge.tail]):
            l = []
            [l.extend(c) for c in children]
            L.append(tuple(l))
    # duplicate children set arise from different splits
    L = set(L)
    for l in L:
        rule = make_rule([node.fi, node.fj, node.ei, node.ej],
                         [[c.fi, c.fj, c.ei, c.ej] for c in l],
                         align.fwords,
                         align.ewords)
        if rule_filter.filter_box(node, l, align):
            rule = make_rule([node.fi, node.fj, node.ei, node.ej],
                             [[c.fi, c.fj, c.ei, c.ej] for c in l],
                             align.fwords,
                             align.ewords)
        #if rule_filter.filter(rule):
            node.rules.append(rule)
            node.children.append(l)

if __name__ == '__main__':
    argv = common.parse_flags()

    ffilename = FLAGS.parallel_corpus[0]
    efilename = FLAGS.parallel_corpus[1]
    afilename = FLAGS.parallel_corpus[2]
    ffile = open(ffilename)
    efile = open(efilename)
    afile = open(afilename)
    alignments = alignment.Alignment.reader_pharaoh(ffile, efile, afile)

    hgs = []

    rule_dumper = RuleDumper()
    for i, a in enumerate(timed(select(alignments)), 1):
        a.write_visual(logger.file)
        #if i != 8:
        #    continue
        #logger.writeln('--- %s ---' % i)
        #a.write_visual(logger.file)
        hg, a = phrase_decomposition_forest(a)
        hgs.append(hg)

        for node in hg.topo_order():
            for edge in node.incoming:
                edge.rule = make_rule([edge.head.fi, edge.head.fj, edge.head.ei, edge.head.ej],
                                      [[x.fi, x.fj, x.ei, x.ej] for x in edge.tail],
                                      a.fwords,
                                      a.ewords)
        #hg.show()

        nnodewithrule = 0
        nrules = 0
        rules = []
        for node in hg.topo_order():
            #print('-- node %s ' % node)
            #for edge in node.incoming:
                #print('- edge %s ' % edge)
                #print([len(t.rules) for t in edge.tail])
            make_composed_rules(node, a)
            #node.make_composed_rules()
            if len(node.rules) > 0:
                nnodewithrule += 1
            #for rule in node.rules:
            #    print(rule)
            #print('%s rules' % len(node.rules))
            rules.extend(node.rules)
            nrules += len(node.rules)
        #print('rules extracted from %s/%s nodes' %
        #      (nnodewithrule, len(hg.nodes)))
        for rule in rules:
            print(rule)
        rule_dumper.add(rules)
        logger.writeln('%s rules extracted from sent %s' % (nrules, i))
        #logger.writeln('%s uniq' % len(set(str(r) for r in rules)))
    rule_dumper.dump()
    ffile.close()
    efile.close()
    afile.close()
