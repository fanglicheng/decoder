import gflags
FLAGS = gflags.FLAGS
from common import cyk_spans
import logger
from lazy_list_merger import LazyListMerger

gflags.DEFINE_integer(
    'glue_span',
    10,
    'Only glue rules are used for spans larger than this.')

class DotItem(object):
    def __init__(self, node, i, j, ants):
        self.node = node  # pointer to grammar trie node
        self.i = i
        self.j = j
        self.ants = ants  # list of items used for deriving this dot item

    # WRONG
    def merge(self, other):
        """Merge other into self. Each bin in self becomes a LazyListMerger
        that behaves the same as a list."""
        assert self == other
        assert len(self.ants) == len(other.ants)
        new_ants = []
        for bin1, bin2 in zip(self.ants, other.ants):
            list_merger = LazyListMerger()
            list_merger.add_list(bin1)
            list_merger.add_list(bin2)
            new_ants.append(list_merger)
        self.ants = tuple(new_ants)

    def __eq__(self, other):
        return (id(self.node) == id(other.node) and
                self.i == other.i and
                self.j == other.j)

    def __hash__(self):
        return hash( (id(self.node), self.i, self.j) )

    def __str__(self):
        return '<DotItem %s,%s: %s>' % (self.i, self.j, self.node.sym_str())

class DotChart(object):
    def __init__(self, chart, grammar):
        self.chart = chart  # finished items chart
        self.grammar = grammar
        self.bins = [[[] for i in range(self.chart.N + 1)]
                     for j in range(self.chart.N + 1)]
        self.index = {}  # index of DotItems
        self.merged = 0  # number of DotItems merged
        # seed
        for i in range(self.chart.N):
            dotitem = DotItem(grammar.root, i, i, ())
            self.add(dotitem)

        if 'glue' in self.grammar.name:
            self.prune = False
        else:
            self.prune = True

    def add(self, dotitem):
        bin = self.bins[dotitem.i][dotitem.j]
        bin.append(dotitem)
        old_dotitem = self.index.get(dotitem)
        if old_dotitem is None:
            self.index[dotitem] = dotitem
        else:
            # old_dotitem.merge(dotitem)
            self.merged += 1

    def expand(self, i, j):
        if not self.prune or j-i <= FLAGS.glue_span:
            self.scan(i, j)
            for k in range(i+1, j):
                self.complete(i, k, j)

    def unary_expand(self, i, j):
        self.complete(i, i, j)

    def scan(self, i, j):
        if logger.level >= 3:
            logger.writeln('Scan: [%s, %s]' % (i, j))
        for dotitem in self.bins[i][j-1]:
            word = self.chart.fwords[j-1]
            next_node = dotitem.node.get(word)
            if next_node is not None:
                new_dotitem = DotItem(next_node, i, j, dotitem.ants)
                if logger.level >= 4:
                    logger.writeln(new_dotitem)
                self.add(new_dotitem)

    def complete(self, i, k, j):
        if logger.level >= 3:
            logger.writeln('Complete: %s %s %s' % (i, k, j))
        for dotitem in self.bins[i][k]:
            for var, bin in self.chart.iter_items_by_nts(k, j):
                next_node = dotitem.node.get(var)
                if next_node is not None:
                    new_dotitem = DotItem(next_node,
                                          i,
                                          j,
                                          dotitem.ants + (bin,))
                    if logger.level >= 4:
                        logger.writeln('new dotitem: %s' % new_dotitem)
                    self.add(new_dotitem)

    def iter_bins(self):
        for i, j in cyk_spans(self.chart.N):
            yield self.bins[i][j]

    def stats(self):
        result = '--DotChart Stats--\n'
        result += 'grammar: %s\n' % (self.grammar.name
                                     if hasattr(self.grammar, 'name')
                                     else '[no name]')
        result += 'DotItems: %s\n' % sum(len(bin) for bin in self.iter_bins())
        result += 'merged: %s\n' % self.merged
        return result
