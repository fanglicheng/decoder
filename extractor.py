#!/usr/bin/env python

# extractor.py
# David Chiang <chiang@isi.edu>

# Copyright (c) 2004-2006 University of Maryland. All rights
# reserved. Do not redistribute without permission from the
# author. Not for commercial use.

# Licheng Fang: Modified to generate an extraction hypergraph.

import sys, os, os.path
import time, math

import gflags
FLAGS = gflags.FLAGS

import rule_extraction_flags
import alignment
from common import timed, select
import logger
from rule import Rule
from phrase_hypergraph import PhraseHGNode, PhraseHGEdge
from hypergraph import Hypergraph
from abc_parser import ABCParser
from lexical_weighter import LexicalWeighter
from rule_dumper import RuleDumper

logger.level = 1

PHRASE = '[A]'

profile = False

if not profile:
    try:
        import psyco
        psyco.profile()
    except ImportError:
        pass

if profile:
    import hotshot, hotshot.stats

# This is a global option because, when True, it allows assumptions
# that are scattered throughout the program
tight_phrases = False

prefix_labels = set()
force_french_prefix = False
force_english_prefix = True

class Extractor(object):
    def __init__(self,
                 maxabslen=10,
                 maxlen=5,
                 minhole=1,
                 maxvars=2,
                 forbid_adjacent=True,
                 require_aligned_terminal=True,
                 tight_phrases=True,
                 remove_overlaps=False,
                 lexical_weighter=None,
                 keep_word_alignments=False,
                 etree=None,
                 etree_labels=False,
                 maximize_derivation=False):
        self.maxabslen = maxabslen
        self.maxlen = maxlen
        self.minhole = minhole
        self.maxvars = maxvars
        self.forbid_adjacent = forbid_adjacent
        self.require_aligned_terminal = require_aligned_terminal
        self.tight_phrases = tight_phrases
        self.remove_overlaps = remove_overlaps
        self.lexical_weighter = lexical_weighter
        self.keep_word_alignments = keep_word_alignments
        self.etree = etree
        self.etree_labels = etree_labels
        self.maximize_derivation = maximize_derivation

    def extract_rules(self, a):
        phrases = self.extract(a)
        rules, phrase_nodes = self.subtract_phrases(a, phrases)
        return rules

    def extract_hypergraph(self, a):
        phrases = self.extract(a)
        rules, phrase_nodes = self.subtract_phrases(a, phrases)
        if len(phrase_nodes) == 0:
            return None
        new_f_len, new_e_len = a.remove_unaligned_words(phrase_nodes)
        parser = ABCParser(new_f_len, new_e_len, phrase_nodes)
        hg = parser.parse_agenda()
        return hg

    # ----------- begin of methods class users usually do not need------------

    def extract(self, a):
        phrases = list(extract_phrases(a, self.maxabslen))
        if not self.tight_phrases:
            phrases = list(loosen_phrases(a, phrases, self.maxabslen))
        if self.remove_overlaps:
            phrases = remove_overlaps(a, phrases)

        # remove words not covered by any phrases. this is required for
        # generating a hypergraph
        # a.write_visual(sys.stdout)
        phrases = a.remove_uncovered_words(phrases)
        if logger.level >= 3:
            logger.writeln()
            logger.writeln(' '.join(str(w) for w in a.fwords))
            logger.writeln(' '.join(str(w) for w in a.ewords))

        self.lexical_weighter.compute_lexical_weights(a)

        # phrase_index = set()
        # ephrase_index = set()
        # for (fi,ei,fj,ej) in phrases:
        #     phrase_index.add((fi,fj))
        #     ephrase_index.add((ei,ej))

        # empty espans suggests an error
        if a.espans is not None and len(a.espans) > 0: 
            #add_multiconstituents(a, maxabslen, ephrase_index, 2)
            #add_constituent_prefixes(a, ephrase_index)
            add_sister_prefixes(a, ephrase_index, self.etree)
            #add_bounded_prefixes(a, ephrase_index, etree)
            #pass

        # filter using English trees
        if a.espans is not None:
            #phrases = list(phrases)
            phrases = [(fi,ei,fj,ej) 
                       for (fi,ei,fj,ej) in phrases
                       if a.espans.has_key((ei,ej))]
            #phrases = random.sample(phrases, len(dummy_phrases))
                
        if logger.level >= 3:
            sys.stderr.write("Initial phrases:\n")
            phrases = list(phrases)
            for (i1,j1,i2,j2) in phrases:
                sys.stderr.write("(%d,%d) %s (%d,%d) %s\n" % \
                                 (i1,
                                  i2,
                                  " ".join([sym.tostring(w) 
                                            for w in a.fwords[i1:i2]]),
                                  j1,
                                  j2,
                                  " ".join([sym.tostring(w) 
                                            for w in a.ewords[j1:j2]])))
        
        # note by Fang:
        # without "relabel with English tree" option set, 
        # all phrases are simply labeled with "[PHRASE]"
        labeled = []
        for (fi,ei,fj,ej) in phrases:
            labeled.extend((x,fi,ei,fj,ej) 
                           for x in self.label_phrase(a, (fi,fj,ei,ej)))
        phrases = labeled
        return phrases

    def subtract_phrases(self, a, phrases):
        if type(phrases) is not list:
            phrases = list(phrases)
        if len(phrases) == 0:
            logger.write("warning: no phrases extracted\n")
            return [], []
        maxabslen = max([i2-i1 for (x,i1,j1,i2,j2) in phrases])
    
        # Search space can be thought of as a directed graph where the
        # vertices are between-symbol positions, and the edges are
        # words or phrases. Our goal is to find all paths in the graph
        # which start at the left edge of a phrase and end at the right edge.
    
        # Find all paths of length maxlen or less starting from the left
        # edge of a phrase
    
        # note by Fang:
        # possible f side phrases are represented as a list of integer indices
        # interspersed with subphrases, which are five-tuples.
    
        n = len(a.fwords)
        chart = [[[[] for nvars in range(self.maxvars+1)]
                  for i2 in range(n+1)]
                 for i1 in range(n+1)]
        i2index = [[] for i2 in range(n+1)]
        i1s = set()
    
        phrase2node = {}
    
        phrase_nodes = []
        for phrase in phrases:
            (x,i1,j1,i2,j2) = phrase
            node = PhraseHGNode(x, i1, i2, j1, j2)
            node.max_d = 0
            phrase_nodes.append(node)
            phrase2node[phrase] = node
            i2index[i2].append(phrase)
            i1s.add(i1)
    
        for i in i1s:
            chart[i][i][0].append([])
        for k in range(1,min(maxabslen,n)+1):
            for i1 in [i for i in i1s if i+k<=n]:
                i2 = i1+k
                # find a subphrase
                for subphrase in i2index[i2]:
                    (sub_x, sub_i1, sub_j1, sub_i2, sub_j2) = subphrase
                    if sub_i2-sub_i1 >= self.minhole: # or sub_x is not PHRASE:
                        # not very efficient because no structure sharing
                        for nvars in range(self.maxvars):
                            for item in chart[i1][sub_i1][nvars]:
                                if (len(item) < self.maxlen
                                    and not (self.forbid_adjacent and len(item)>0 and type(item[-1]) is not int)
                                    # force prefix categories to be at left edge of French side
                                    and not (force_french_prefix and len(item) > 0 and sub_x in prefix_labels)
                                    ):
                                        chart[i1][i2][nvars+1].append(item + [subphrase])
                # find a word
                # note by Fang:
                # this extends all the possible rules one step down the word sequence
                for nvars in range(self.maxvars+1):
                    for item in chart[i1][i2-1][nvars]:
                        if len(item) < self.maxlen:
                            chart[i1][i2][nvars].append(item + [i2-1])
    
        # Now for each phrase, find all paths starting from left edge and
        # ending at right edge
    
        wholeresult = []
        if self.maximize_derivation:
            phrases.sort(key=lambda p: p[3]-p[1]) # sort by length
        for phrase in phrases:
            x, i1, j1, i2, j2 = phrase
            node = phrase2node[phrase]
            result = []
            for nvars in range(self.maxvars+1):
                for fwords in chart[i1][i2][nvars]:
                    # note by Fang:
                    # all the f side possibilities are fed into "make_rule",
                    # and whether the e side can form a rule is judged in 
                    # "make_rule".
                    r = self.make_rule(a, (x,i1,j1,i2,j2), fwords)
                    if r is not None:
                        # intermediate rule scoring: [count, lex_e2f, lex_f2e]
                        r.feats = [1.0]
                        r.feats.extend(self.lexical_weighter.score_rule(a, r))
                        result.append(r)
    
                        d_len = 1
                        edge = PhraseHGEdge(r)
                        for fi in fwords:
                            if type(fi) is not int:
                                tailnode = phrase2node[fi]
                                edge.add_tail(tailnode)
                                d_len += tailnode.max_d
                        if self.maximize_derivation:
                            if d_len == node.max_d:
                                node.add_incoming(edge)
                            elif d_len > node.max_d:
                                node.incoming = []
                                node.max_d = d_len
                                node.add_incoming(edge)
                        else:
                            node.add_incoming(edge)
            # Normalize
            for r in result:
                r.feats = [x/len(result) for x in r.feats]
            wholeresult.extend(result)

        return wholeresult, phrase_nodes

    def label_phrase(self, a, phrase):
        fi, fj, ei, ej = phrase
        if self.etree_labels and a.espans is not None:
            return a.espans.get((ei,ej), [PHRASE])
        if self.etree_labels:
            sys.stderr.write("this shouldn't happen either (line %d)\n" % 
                             a.lineno)
        return [PHRASE]

    def make_rule(self, a, source_phrase, fwords):
        '''fwords is a list of numbers and subphrases:
           the numbers are indices into the French sentence

           note by Fang: the input for make_rule is an initial phrase and a
           possible rule construction, which is plausible only for the f side
           at this moment. 'make_rule' ensures that the e sides of the
           subphrases fit into the initial phrase being subtracted and don't
           overlap.  the outputed rule object includes information of the
           lexicalized symbols at both sides, their indices into the original
           sentence pair (fpos, epos), and possibly the word alignment info.
           ''' 
        x, i1, j1, i2, j2 = source_phrase

        # omit trivial rules
        if len(fwords) == 1 and type(fwords[0]) is not int:
            return None
    
        #if not tight_phrases:
        fwords = fwords[:]
        fpos = [None for w in fwords] # map from index in phrase to index in sentence
        ewords = a.ewords[j1:j2]
        elen = j2-j1
        index = 0  # nonterminal index
        flag = False
        for i in range(len(fwords)):
            fi = fwords[i]
            if type(fi) is int: # terminal symbol
                if a.faligned[fi]:
                    flag = True
                fwords[i] = a.fwords[fi]
                fpos[i] = fi
            else: # nonterminal symbol
                (sub_x,sub_i1,sub_j1,sub_i2,sub_j2) = fi
                sub_j1 -= j1
                sub_j2 -= j1
                
                if not tight_phrases:
                    # Check English holes
                    # can't lie outside phrase
                    if sub_j1 < 0 or sub_j2 > elen:
                        return None
    
                    # can't overlap
                    for j in range(sub_j1,sub_j2):
                        if type(ewords[j]) is tuple or ewords[j] is None:
                            return None
    
                # Set first eword to var, rest to None
    
                # We'll clean up the Nones later
                v = sub_x
                fwords[i] = v
                fpos[i] = (sub_i1,sub_i2)
                ewords[sub_j1] = (v, index, sub_j1+j1, sub_j2+j1)
                for j in range(sub_j1+1,sub_j2):
                    ewords[j] = None
                index += 1
    
        # Require an aligned French word
        if self.require_aligned_terminal and not flag:
            return None
    
        epos = []
        new_ewords = []
        e2f = []
        for i in range(elen):
            if ewords[i] is not None:
                if type(ewords[i]) is tuple:
                    (v, index, ei, ej) = ewords[i]
                    # force slash categories to be at left edge of English side
                    #if force_english_prefix and len(new_ewords) != 0 and sym.clearindex(v) in prefix_labels:
                    #    return None
                    e2f.append(index)
                    new_ewords.append(v)
                    epos.append((ei,ej))
                else:
                    new_ewords.append(ewords[i])
                    epos.append(i+j1)
    
                    
        #r = XRule(x,rule.Phrase(tuple(fwords)), rule.Phrase(tuple(new_ewords)))
        r = Rule()
        r.lhs = PHRASE
        r.f = fwords
        r.e = new_ewords
        r.e2f = e2f
        r.fpos = fpos
        r.epos = epos
        r.span = (i1,i2,j1,j2)
    
        if self.keep_word_alignments:
            r.word_alignments = []
            for fi in range(len(fpos)):
                if type(fpos[fi]) is int:
                    for ei in range(len(epos)):
                        if type(epos[ei]) is int:
                            if a.aligned[fpos[fi]][epos[ei]]:
                                r.word_alignments.append((fi,ei))
        #print(r)
        return r

def extract_phrases(self, maxlen=1000000000):
    ifirst = [len(self.fwords) for j in self.ewords]
    ilast = [0 for j in self.ewords]
    jfirst = [len(self.ewords) for i in self.fwords]
    jlast = [0 for i in self.fwords]
    for i in range(len(self.fwords)):
        for j in range(len(self.ewords)):
            if self.aligned[i][j]:
                if j<jfirst[i]:
                    jfirst[i] = j
                jlast[i] = j+1
                if i<ifirst[j]:
                    ifirst[j] = i
                ilast[j] = i+1

    for i1 in range(len(self.fwords)):
        if not self.faligned[i1]:
            continue
        j1 = len(self.ewords)
        j2 = 0
        for i2 in range(i1+1,min(len(self.fwords),i1+maxlen)+1):
            if not self.faligned[i2-1]:
                continue
            # find biggest empty left and right blocks
            j1 = min(j1, jfirst[i2-1])
            j2 = max(j2, jlast[i2-1])

            # make sure band isn't empty
            if j1 >= j2:
                continue

            # check minimum top and bottom blocks
            if j2-j1 > maxlen:
                break # goto next i1 value

            next = 0
            for j in range(j1, j2):
                if ifirst[j]<i1:
                    next = 1
                    break
                if ilast[j]>i2:
                    next = 2
                    break
            if next == 1:
                break # goto next i1 value
            elif next == 2:
                continue # goto next i2 value

            yield((i1,j1,i2,j2))

def loosen_phrases(self, phrases, maxlen):
    for (i1_max,j1_max,i2_min,j2_min) in phrases:
        i1_min = i1_max
        while i1_min > 0 and self.faligned[i1_min-1] == 0:
            i1_min -= 1
        j1_min = j1_max
        while j1_min > 0 and self.ealigned[j1_min-1] == 0:
            j1_min -= 1
        i2_max = i2_min
        while i2_max < len(self.fwords) and self.faligned[i2_max] == 0:
            i2_max += 1
        j2_max = j2_min
        while j2_max < len(self.ewords) and self.ealigned[j2_max] == 0:
            j2_max += 1

        for i1 in range(i1_min, i1_max+1):
            for i2 in range(max(i1+1,i2_min), min(i2_max,i1+maxlen)+1):
                for j1 in range(j1_min, j1_max+1):
                    for j2 in range(max(j1+1,j2_min), min(j2_max,j1+maxlen)+1):
                        yield (i1,j1,i2,j2)

def remove_overlaps(self, phrases):
    # Give priority to leftmost phrases. This yields a left-branching structure
    phrases = [(i1,i2,j1,j2) for (i1,j1,i2,j2) in phrases]
    phrases.sort()
    newphrases = []
    for (i1,i2,j1,j2) in phrases:
        flag = True
        for (i3,j3,i4,j4) in newphrases:
            if i1<i3<i2<i4 or i3<i1<i4<i2: # or j1<j3<j2<j4 or j3<j1<j4<j2:
                flag = False
                break
        if flag:
            newphrases.append((i1,j1,i2,j2))
    return newphrases

# not used
def can_binarize(a, r, phrase_index):
    """can you quasi-binarize this rule while respecting the word alignment?"""
    if r.arity() <= 2:
        return 1
    if r.arity() > 3:
        raise ValueError("4-ary rules and above not supported yet")

    fvars = [x for x in r.fpos if type(x) is tuple]
    for (fi,fj) in phrase_index:
        if fi <= fvars[0][0] and fvars[1][1] <= fj <= fvars[2][0]:
            return 1
        if fvars[0][1] <= fi <= fvars[1][0] and fvars[2][1] <= fj:
            return 1

    return 0

# not used
def compute_cumulative(a):
    a.cumul = [[None for ei in range(len(a.ewords)+1)] for fi in range(len(a.fwords)+1)]
    for fi in range(len(a.fwords)+1):
        a.cumul[fi][0] = 0
    for ei in range(len(a.ewords)+1):
        a.cumul[0][ei] = 0
    for fi in range(1,len(a.fwords)+1):
        for ei in range(1,len(a.ewords)+1):
            a.cumul[fi][ei] = a.cumul[fi][ei-1] + a.cumul[fi-1][ei] - a.cumul[fi-1][ei-1]
            if a.aligned[fi-1][ei-1]:
                a.cumul[fi][ei] += 1

def add_multiconstituents(a, maxabslen, ephrase_index, consts):
    elen = len(a.ewords)
    
    chart = [[None for ej in range(elen+1)] for ei in range(elen+1)]
    for ((ei,ej),labels) in a.espans.iteritems():
        chart[ei][ej] = [labels[0]] # take the highest label

    for el in range(2,maxabslen+1):
        for ei in range(elen-el+1):
            ej = ei+el
            if chart[ei][ej] is not None: # must be a singleton
                continue
            bestsplit = None
            bestlen = None
            for ek in range(ei+1,ej):
                if chart[ei][ek] is not None and chart[ek][ej] is not None and (bestlen is None or len(chart[ei][ek])+len(chart[ek][ej]) < bestlen):
                    bestsplit = ek
                    bestlen = len(chart[ei][ek])+len(chart[ek][ej])
            if bestlen is not None and bestlen <= consts:
                chart[ei][ej] = chart[ei][bestsplit]+chart[bestsplit][ej]
    for (ei,ej) in ephrase_index:
        if not a.espans.has_key((ei,ej)) and chart[ei][ej] is not None:
            a.espans[ei,ej] = [sym.fromtag("_".join(sym.totag(x) for x in chart[ei][ej]))]

def add_constituent_prefixes(a, ephrase_index):
    """if a phrase is a prefix of a constituent, give it a fake label"""
    if logger.level >= 3:
        logger.write(str([(i,j,sym.tostring(x)) for ((i,j),l) in a.espans.iteritems() for x in l ]))
        logger.write("\n")
    
    ei_index = {}
    for ((ei,ej),labels) in a.espans.iteritems():
        ei_index.setdefault(ei, []).extend([(ej,x) for x in reversed(labels)])
    for ei in ei_index.iterkeys():
        ei_index[ei].sort() # stable
        
    for (ei,ej) in ephrase_index:
        if True or not (a.espans.has_key((ei,ej)) and len(a.espans[ei,ej]) > 0):
            for (ej1,x) in ei_index.get(ei,[]):
                if ej1 > ej:
                    x1 = sym.fromtag(sym.totag(x)+"*")
                    a.espans.setdefault((ei,ej),[]).append(x1)
                    prefix_labels.add(x1)
                    break

    if logger.level >= 3:
        logger.write(str([(i,j,sym.tostring(x)) for ((i,j),l) in a.espans.iteritems() for x in l ]))
        logger.write("\n---\n")

def remove_req(node):
    parts = node.label.split("-")
    if parts[-1] in ["HEAD", "C", "REQ"]:
        parts[-1:] = []
        node.required = True
    else:
        node.required = False
    node.label = "-".join(parts)
    for child in node.children:
        remove_req(child)

def add_sister_prefixes_helper(a, ephrases, enode, i):
    """if a phrase comprises one or more (but not all) leftmost children of a constituent, then add it and give it a fake label"""

    j = i+enode.length
    if logger.level >= 3:
        logger.write("(i,j) = %s\n" % ((i,j),))
    x = enode.label
    j1 = i
    for ci in range(len(enode.children)):
        child = enode.children[ci]
        j1 += child.length
        if logger.level >= 3:
            logger.write("(i,j1) = %s\n" % ((i,j1),))
        if j1 < j and (i,j1) in ephrases:

            # constprefix3:
            #x1 = sym.fromtag("%s*" % x)

            # subcat-lr2:
            #subcat = [sister.label for sister in enode.children[ci+1:] if sister.required]
            #x1 = sym.fromtag("/".join(["%s*"%x]+subcat))
            
            # markov1:
            x1 = sym.fromtag("%s/%s" % (x, enode.children[ci+1].label))

            # markov2:
            #x1 = sym.fromtag("%s(%s)" % (x, enode.children[ci].label))
            
            a.espans.setdefault((i,j1),[]).append(x1)
            prefix_labels.add(x1)
            
    for child in enode.children:
        add_sister_prefixes_helper(a, ephrases, child, i)
        i += child.length

def add_sister_prefixes(a, ephrases, etree):
    if logger.level >= 3:
        logger.write("phrases before filtering:\n")
        for (i,j) in ephrases:
            logger.write("%s" % ((i,j),))
        logger.write("constituents before adding:\n")
        for ((i,j),l) in a.espans.iteritems():
            logger.write("%s %s\n" % ((i,j),[sym.tostring(x) for x in l]))

    add_sister_prefixes_helper(a, ephrases, etree, 0)

    if logger.level >= 3:
        logger.write("constituents after adding:\n")
        for ((i,j),l) in a.espans.iteritems():
            logger.write("%s %s\n" % ((i,j),[sym.tostring(x) for x in l]))
        logger.write("\n---\n")

def add_bounded_prefixes_helper(a, phrases, node, i, stack):
    j = i+node.length
    if node.label in ['NP']:
        stack = stack+[(node.label,i,j)]
    else:
        stack = [(node.label,i,j)]
    i1 = i
    for child in node.children:
        if i1 > i:
            for (x,i0,j0) in stack:
                if (i0,i1) in phrases:
                    x1 = sym.fromtag("%s*" % x)
                    a.espans.setdefault((i0,i1),[]).append(x1)
                    prefix_labels.add(x1)
        add_bounded_prefixes_helper(a, phrases, child, i1, stack)
        i1 += child.length

def add_bounded_prefixes(a, ephrases, etree):
    if logger.level >= 3:
        logger.write(str([(i,j,sym.tostring(x)) for ((i,j),l) in a.espans.iteritems() for x in l ]))
        logger.write("\n")

    add_bounded_prefixes_helper(a, ephrases, etree, 0, [])

    if logger.level >= 3:
        logger.write(str([(i,j,sym.tostring(x)) for ((i,j),l) in a.espans.iteritems() for x in l ]))
        logger.write("\n---\n")

def main():
    import gc
    gc.set_threshold(100000,10,10) # this makes a huge speed difference
    #gc.set_debug(gc.DEBUG_STATS)

    input_file = open(FLAGS.parallel_corpus[2])

    if FLAGS.hypergraph is not None:
        try:
            os.mkdir(FLAGS.hypergraph)
        except OSError:
            sys.stderr.write("warning: directory %s already exists\n" %
                             FLAGS.hypergraph)

    ffilename = FLAGS.parallel_corpus[0]
    efilename = FLAGS.parallel_corpus[1]
    ffile = open(ffilename)
    efile = open(efilename)

    if FLAGS.weightfiles is not None:
        fweightfile, eweightfile = FLAGS.weightfiles
    else:
        fweightfile = None
        eweightfile = None

    lexical_weighter = LexicalWeighter(fweightfile,
                                       eweightfile)

    maxlen = FLAGS.maxlen
    maxabslen = FLAGS.maxabslen
    tight_phrases = FLAGS.tight

    prev_time = start_time = time.time()
    slice = 1000

    if profile:
        prof = hotshot.Profile("extractor.prof")
        prof.start()

    if logger.level >= 1:
        sys.stderr.write("(2) Extracting rules\n")
    count = 1
    realcount = 0
    slice = 1000
    if FLAGS.pharaoh:
        alignments = alignment.Alignment.reader_pharaoh(ffile, efile, input_file)
    else:
        alignments = alignment.Alignment.reader(input_file)
        # bug: ignores -W option

    rule_dumper = RuleDumper()

    for i, a in enumerate(select(alignments), 1):
        a.lineno = count

        if logger.level >= 2:
            a.write(logger.file)
            a.write_visual(logger.file)

        etree = None

        # done reading all input lines
        realcount += 1

        extractor = Extractor(maxabslen,
                              maxlen,
                              FLAGS.minhole,
                              FLAGS.maxvars,
                              FLAGS.forbid_adjacent,
                              FLAGS.require_aligned_terminal,
                              tight_phrases,
                              FLAGS.remove_overlaps,
                              lexical_weighter,
                              FLAGS.keep_word_alignments,
                              etree,
                              FLAGS.etree_labels)
        rules = extractor.extract_rules(a)

        if logger.level >= 3:
            sys.stderr.write("Rules:\n")
            rules = list(rules)
            for r in rules:
                sys.stderr.write("%d ||| %s\n" % (realcount, r))

        if False:
            rules = list(rules)
            for r in rules:
                sys.stderr.write("%d ||| %s ||| %f %f\n" % (realcount-1, r, r.scores[1]/r.scores[0], r.scores[2]/r.scores[0]))

        #logger.writeln('%s rules extracted from sent %s' % (len(rules), i))

        rule_dumper.add(rules)

        if logger.level >= 1 and count%slice == 0:
            sys.stderr.write("time: %f, sentences in: %d (%.1f/sec), " % (time.time()-start_time, count, slice/(time.time()-prev_time)))
            sys.stderr.write("rules out: %d+%d\n" % (rule_dumper.dumped, len(rule_dumper.gram)))
            prev_time = time.time()

        count += 1

    rule_dumper.dump()

    if profile:
        prof.stop()
        prof.close()
        stats = hotshot.stats.load("extractor.prof")
        stats.strip_dirs()
        stats.sort_stats('time', 'calls')
        stats.print_stats(100)
    
if __name__ == "__main__":
    try:
        argv = FLAGS(sys.argv)  # parse flags
    except gflags.FlagsError as e:
        print('%s\nUsage: %s ARGS\n%s' % (e, sys.argv[0], FLAGS))
        sys.exit(1)

    main()
