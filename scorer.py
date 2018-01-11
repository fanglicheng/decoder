#!/usr/bin/env python
# (note that 32-bit Python uses less memory)

# scorer.py
# David Chiang <chiang@isi.edu>

# Copyright (c) 2004-2006 University of Maryland. All rights
# reserved. Do not redistribute without permission from the
# author. Not for commercial use.

import sys, os, os.path
import time, math
import heapq

import gflags
FLAGS = gflags.FLAGS
from extractor import PHRASE
from rule import Rule, isvar
from logprob import elog10
import logger
import translation_job

gflags.DEFINE_string(
    'persent',
    None,
    'Per-sentence filtering output directory.')
gflags.DEFINE_list(
    'unfiltered',
    None,
    'Files and directories containing unfiltered rules.')
gflags.DEFINE_string(
    'filtered',
    'filtered.hiero.gr',
    'Output filename for filtered grammar')
gflags.DEFINE_string(
    'filter_file',
    None,
    'Test file to filter for.')
gflags.DEFINE_integer(
    'filter_maxlen',
    5,
    'Maximum number of terminals and nonterminals in filtering.')
gflags.DEFINE_float(
    'count_cut',
    0.0,
    'Rules with count lower than this number are thrown away.')
gflags.DEFINE_integer(
    'discard_raw_count',
    0,
    'Discard rules that occur fewer than this number. \
     Works only when --noaccumulate is set for extractor.')

logger.level = 1

profile = False

if not profile:
    try:
        import psyco
        psyco.profile()
    except ImportError:
        pass

if profile:
    import hotshot, hotshot.stats

import bisect
class Filter(object):
    def __init__(self, sents, maxlen):
        self.all_substrs = {} # could replace with a suffix array for compactness
        self.substrs = [{} for sent in sents]
        self.n = [len(sent) for sent in sents]
        self.allsents = [i for i in range(len(sents))]  # return this when all sents match
        self.maxlen = maxlen
        for sent_i in range(len(sents)):
            sent = sents[sent_i]
            for i in range(len(sent)):
                for j in range(i+1, min(i+maxlen,len(sent))+1):
                    self.all_substrs.setdefault(tuple(sent[i:j]), set()).add(sent_i)
                    #self.all_substrs.setdefault(tuple(sent[i:j]), []).append(sent_i)
                    self.substrs[sent_i].setdefault(tuple(sent[i:j]), []).append(i)
            for positions in self.substrs[sent_i].values():
                positions.sort()
        for (key,sents) in self.all_substrs.items():
            self.all_substrs[key] = list(sents)
            self.all_substrs[key].sort()

    def match1(self, pattern, substrs, n):
        #note by Fang:
        #pattern is a phrase object
        pos = 0
        for chunk in self.get_chunks(pattern):
            if len(chunk) == 0: # very common case
                if pos > n:
                    return False # even empty chunk can't match past end of sentence
                pos += 1 # +1 for hole filler
                continue
            if not chunk in substrs:
                return False
            positions = substrs[chunk]
            i = bisect.bisect_left(positions, pos)
            if i < len(positions):
                pos = positions[i]+len(chunk)+1 # +1 for hole filler
            else:
                return False
        return True

    def match(self, rule):
        pattern = rule.f
        if rule.arity == 0: # very common case
            return tuple(pattern) in self.all_substrs

        sentlists = []
        for chunk in self.get_chunks(pattern):
            if len(chunk) > 0:  # there are zero length chunks
                sentlist = self.all_substrs.get(chunk, None)
                if sentlist is None:
                    return False
                sentlists.append(sentlist)

        n = len(sentlists)
        if n == 0:  # rule contains no terminals
            return True
        elif n == 1: # common case
            for sentnum in sentlists[0]:
                if self.match1(pattern, self.substrs[sentnum], self.n[sentnum]):
                    return True
            return False

        indices = [0]*n
        sentnum = max(sentlists[i][0] for i in range(n))
        while True:
            for i in range(n):
                if sentlists[i][indices[i]] < sentnum:
                    indices[i] = bisect.bisect_left(sentlists[i], sentnum)
                    if indices[i] >= len(sentlists[i]):
                        return False
                    if sentlists[i][indices[i]] > sentnum:
                        sentnum = sentlists[i][indices[i]]
                        break
            else: # all sentence numbers match
                if self.match1(pattern, self.substrs[sentnum], self.n[sentnum]):
                    return True
                else:
                    sentnum += 1
        return False

    def match2(self, rule):
        "lfang: return a list of matching sentences."
        pattern = rule.f
        if rule.arity == 0: # very common case
            return self.all_substrs.get(tuple(pattern), [])

        sentlists = []
        for chunk in self.get_chunks(pattern):
            if len(chunk) > 0:  # there are zero length chunks (adjacent nonterminals)
                sentlist = self.all_substrs.get(chunk, None)
                if sentlist is None:
                    return []
                sentlists.append(sentlist)

        n = len(sentlists)
        if n == 0:  # rule contains no terminals
            return self.allsents
        elif n == 1: # common case
            return [sentnum for sentnum in sentlists[0]
                    if self.match1(pattern, self.substrs[sentnum], self.n[sentnum])]

        indices = [0]*n
        sentnum = max(sentlists[i][0] for i in range(n))
        matches = []
        while True:
            for i in range(n):
                if sentlists[i][indices[i]] < sentnum:
                    indices[i] = bisect.bisect_left(sentlists[i], sentnum)
                    if indices[i] >= len(sentlists[i]):
                        return matches
                    if sentlists[i][indices[i]] > sentnum:
                        sentnum = sentlists[i][indices[i]]
                        break
            else: # all sentence numbers match
                if self.match1(pattern, self.substrs[sentnum], self.n[sentnum]):
                    matches.append(sentnum)
                sentnum += 1

    def get_chunks(self, pattern):
        """Iterator for chunks (contiguous terminals) in a phrase. If there are
        N nonterminals in the phrase, get_chunks returns exactly N+1 chunks,
        including zero-length ones."""
        chunk = []
        for sym in pattern:
            if isvar(sym):
                yield tuple(chunk)
                chunk = []
            else:
                chunk.append(sym)
        yield tuple(chunk)

n_rule_discarded = 0
### rule reading

def read_rules(files):
    """Merge several grammar files together (assuming they are sorted).
    If duplicate rules are found, their scores are added."""
    # note by Fang:
    # the heap is initialized with one line from each file, whenever a line
    # from a certain file is popped out, another line is read from the same
    # file. 'sumrule' accumulates scores from duplicated rules until a
    # different rule is encountered, when sumrule is yielded.  counting of rule
    # occurrences is done here, counting of e side and f side is done in
    # 'tabulate'
    global n_rule_discarded
    heap = []
    fid2f = {}
    for f in files:
        fid2f[id(f)] = f
    for f in files:
        try:
            line = next(f)
        except StopIteration:
            pass
        else:
            heap.append((line, id(f)))
    heapq.heapify(heap)

    savehandle = sumrule = None
    n_sum = 0
    while len(heap) > 0:
        (line, fid) = heapq.heappop(heap)
        f = fid2f[fid]
        # assume rule1 == rule2 => handle1 == handle2
        (handle, ruleline) = line.split("|||", 1)

        try:
            r = Rule()
            r.fromstr(ruleline)
        except:
            sys.stderr.write("couldn't scan rule: %s\n" % ruleline)
            r = None

        if len(r.feats) < 1:
            sys.stderr.write("rule doesn't have enough scores: %s\n" % str(r))
            r = None

        if r is not None:
            if sumrule is not None and r == sumrule:
                sumrule.feats = [f1 + f2 
                                 for f1, f2 in zip(sumrule.feats, r.feats)]
                n_sum += 1
            else:
                if sumrule is not None and sumrule.feats[0] >= FLAGS.count_cut:
                    if n_sum > FLAGS.discard_raw_count:
                        yield savehandle, sumrule
                    else:
                        n_rule_discarded += 1
                sumrule = r
                savehandle = handle
                n_sum = 1
        try:
            line = next(f)
        except StopIteration:
            pass
        else:
            heapq.heappush(heap, (line, id(f)))
    if sumrule is not None and sumrule.feats[0] >= FLAGS.count_cut:
        if n_sum > FLAGS.discard_raw_count:
            yield savehandle, sumrule
        else:
            n_rule_discarded += 1

def read_rule_blocks(files):
    """Read all the rules with the same English side at a time, assuming
    that they are coming in English-major order."""
    block = None
    prev_handle = None
    for (handle, r) in read_rules(files):
        if prev_handle is not None and handle == prev_handle:
            block.append(r)
        else:
            if prev_handle is not None:
                yield block
            block = [r]
            prev_handle = handle
    if block is not None:
        yield block

def get_rules(inputs):
    "Return rule blocks. An input can be either a file or a dir."
    inputfiles = []
    for input in inputs:
        if os.path.isdir(input):
            inputfiles.extend(os.path.join(input, name) for name in os.listdir(input))
        else:
            inputfiles.append(input)
    inputfiles = [open(inputfile) for inputfile in inputfiles]
    for block in read_rule_blocks(inputfiles):
        yield block

interval = 100000

class RuleCache(object):
    def __init__(self, filename, size):
        self.size = size
        self.filename = filename
        self.rules = []

    def add(self, rule):
        self.rules.append(rule)
        if len(self.rules) > self.size:
            self.dump()
            self.rules = []

    def dump(self):
        if len(self.rules) > 0:
            f = open(self.filename, 'a')
            for rule in self.rules:
                f.write('%s\n' % rule)
            f.close()

class Tabulator(object):
    def __init__(self, ffilter, outputdir):
        self.ffilter = ffilter
        self.fsum = {}
        self.xsum = {}
        self.n = len(ffilter.n)  # number of sentences in filter set
        self.outputdir = outputdir
        os.system('rm -rf %s' % outputdir)
        os.mkdir(outputdir)
        self.cachesize = 1000000
        self.cache = [RuleCache(self.grammarfilename(i),
                                self.cachesize/self.n)
                      for i in range(self.n)]
        self.dumpfile = os.path.join(self.outputdir, 'filtered.tmp')

        self.rulesin = 0
        self.rulesout = 0

    def run(self):
        self.tabulate()
        self.calculate()

    def tabulate(self):
        global start_time
        f = open(self.dumpfile, 'w')
        start_time = time.time()
        for rules in get_rules(FLAGS.unfiltered):
            esum = 0.0
            for r in rules:
                esum += r.feats[0]
            for r in rules:
                self.rulesin += 1
                self.xsum[r.lhs] = self.xsum.get(r.lhs, 0.0) + r.feats[0]
                sentlist = self.ffilter.match2(r)
                weight = r.feats[0]
                r.feats = [esum] + r.feats
                if sentlist:
                    self.rulesout += 1
                    fhandle = tuple(r.f)
                    self.fsum[fhandle] = self.fsum.get(fhandle, 0.0) + weight
                    # save esum to disk for later use
                    f.write('%s ||||| %s\n' %
                            (' '.join(str(x) for x in sentlist), r))
                if logger.level >= 1 and self.rulesin % interval == 0:
                    sys.stderr.write("time: %f, rules in: %d, rules out: %d\n" %
                                     (time.time()-start_time,
                                      self.rulesin,
                                      self.rulesout))
        f.close()

    def calculate(self):
        global start_time
        start_time = time.time()
        f = open(self.dumpfile)
        count = 0
        for line in f:
            sentlist, rulestr = line.split('|||||')
            sentlist = [int(x) for x in sentlist.split()]
            r = Rule()
            r.fromstr(rulestr)
            r.feats = feature_vector(r.feats[1],
                                     self.xsum[r.lhs],
                                     r.feats[0],
                                     self.fsum[tuple(r.f)],
                                     r.feats[2:4],
                                     r.feats[4:])
            for i in sentlist:
                self.cache[i].add(r)
            count += 1
            if logger.level >= 1 and count % interval == 0:
                sys.stderr.write('time: %s, rules out: %d\n' %
                                 (time.time() - start_time,
                                  count))
        f.close()
        for c in self.cache:
            c.dump()

    def grammarfilename(self, i):
        return os.path.join(self.outputdir, 'g' + str(i+1).rjust(6, '0'))

fsum = {} # c(lhs, french)
esum = {} # c(lhs, english)
allsum = 0.0 # c(*)
xsum = {} # c(lhs)
gram = {}

def tabulate():
    if logger.level >= 1:
        sys.stderr.write("(3) Tabulating filtered phrases\n")
    count = 1

    global fsum, esum, allsum, xsum, gram

    # read in all rules with matching english sides at the same time.
    # this way, we can sum only those english sides that ever appeared
    # with a french side that passes the filter.

    #note by Fang:
    #for each rule (f,e) we want need the counts c(f), c(e), and c(f,e),
    #which correspond to fsum, esum, and gram.
    #so if a (f,e) doesn't have an unfiltered f, we don't need to count it, and also,
    #if an e doesn't appear together with any unfiltered f, it's safe to not count it.
    #without sorting the e side first, we cannot know whether a given e will ever appear with
    #an unfiltered f.
    for rules in get_rules(FLAGS.unfiltered):
        flag = False
        for r in rules:
            scores = r.feats
            weight = scores[0]
            allsum += weight
            xsum[r.lhs] = xsum.get(r.lhs, 0.0) + weight
            if ffilter is None or ffilter.match(r): # there used to be a shortcut here -- if fsum.has_key(r.f)
                #fsum[(r.lhs,r.f)] = fsum.get((r.lhs,r.f), 0.0) + weight
                fhandle = tuple(r.f)
                fsum[fhandle] = fsum.get(fhandle, 0.0) + weight
                gram[r] = r
                flag = True
            if logger.level >= 1 and count%interval == 0:
                sys.stderr.write("time: %f, rules in: %d, rules out: %d\n" % (time.time()-start_time, count, len(gram)))

            count += 1
        if flag:
            # ignore the ordering of nonterminals in counting the english side
            ewordsnorm = tuple(rules[0].e)
            #for r in rules:
            #    esum[(r.lhs, ewordsnorm)] = 0.0
            esum[ewordsnorm] = 0.0
            for r in rules:
                scores = r.feats
                #esum[(r.lhs, ewordsnorm)] += scores[0]
                esum[ewordsnorm] += scores[0]

def feature_vector(weight, xsum, esum, fsum, lexs=[], others=[]):
    newscores = [
        -elog10(float(weight)/xsum),             # p(e,f|x)
        #-elog10(float(weight)/esum[(r.lhs, ewordsnorm)]), # p(f|e,x)
        -elog10(float(weight)/esum), # p(f|e)
        #-elog10(float(weight)/fsum[(r.lhs, r.f)]),        # p(e|f,x)
        -elog10(float(weight)/fsum)        # p(e|f)
        #-elog10(float(fsum[r.f])/allsum),          # p(f)
        #-elog10(float(esum[ewordsnorm])/allsum),    # p(e)
        ]
    # the rest of the fields we find the weighted average of, using the first field as weight
    # fields 2 and 3 are interpreted as probabilities, the rest as costs. this is ugly

    # note by Fang:
    # the lexical weights are not supposed to be accumulated. by
    # dividing by weights (no. of occurrences) here, we choose the
    # average lexical scores of rule occurrences with possibly
    # different word alignment inside.
    newscores.extend([-elog10(lex/weight) for lex in lexs])  # P(f|e) and P(e|f)
    newscores.extend([score/weight for score in others])
    return newscores

def calculate():
    if logger.level >= 1:
        sys.stderr.write("(4) Calculating probabilities\n")

    count = 1
    for r in gram:
        if r.feats[0] == 0.0:
            continue
        try:
            r.feats = feature_vector(r.feats[0],
                                     xsum[r.lhs],
                                     esum[tuple(r.e)],
                                     fsum[tuple(r.f)],
                                     r.feats[1:3],
                                     r.feats[3:])
            output_file.write("%s\n" % r)
        except OverflowError:
            sys.stderr.write("warning: division by zero or log of zero: %s\n" % r.to_line())
        if logger.level >= 1 and count%interval == 0:
            sys.stderr.write("time: %f, rules in/out: %d\n" % (time.time()-start_time, count))
        count += 1

ffilter = None
output_file = None
start_time = None

def main():
    global ffilter
    global output_file
    global start_time
    import gc
    gc.set_threshold(100000,10,10) # this makes a huge speed difference
    #gc.set_debug(gc.DEBUG_STATS)

    output_file = open(FLAGS.filtered, 'w')

    if FLAGS.filter_file is not None:
        lines = open(FLAGS.filter_file).readlines()

        if FLAGS.preprocess:
            lines = [translation_job.preprocess(line)[0] for line in lines]

        l = [line.split() for line in lines]
        ffilter = Filter(l, FLAGS.filter_maxlen)
        logger.write("Filtering using %s\n" % FLAGS.filter_file)
    else:
        ffilter = None

    prev_time = start_time = time.time()

    if profile:
        prof = hotshot.Profile("scorer.prof")
        prof.start()

    if FLAGS.persent is None:
        tabulate()
        del ffilter
        calculate()
    else:
        assert ffilter is not None
        tabulator = Tabulator(ffilter, FLAGS.persent)
        tabulator.run()

    if profile:
        prof.stop()
        prof.close()
        stats = hotshot.stats.load("scorer.prof")
        stats.strip_dirs()
        stats.sort_stats('time', 'calls')
        stats.print_stats(100)

    output_file.close()

    logger.write('Rule discarded: %s' % n_rule_discarded)
    logger.write("\nDone\n")

if __name__ == "__main__":
    try:
        argv = FLAGS(sys.argv)  # parse flags
    except gflags.FlagsError as e:
        print('%s\nUsage: %s ARGS\n%s' % (e, sys.argv[0], FLAGS))
        sys.exit(1)

    main()
