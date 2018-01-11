#!/usr/bin/env python3

from heapq import heapify, heappush, heappop

import gflags
FLAGS = gflags.FLAGS
import logger
from common import INF
from rule import is_virtual, nocat
import decoding_flags

gflags.DEFINE_boolean(
    'ban_negative_deduction',
    False,
    'Ban deductions with negative costs during decoding.')
gflags.DEFINE_string(
    'glue_var',
    '[A]',
    'Glue nonterminal.')
gflags.DEFINE_string(
    'straight_var',
    '[STRAIGHT]',
    'Straight nonterminal.')
gflags.DEFINE_string(
    'inverted_var',
    '[INVERTED]',
    'Inverted nonterminal.')

class Bin(object):
    def __init__(self, K, chart):
        self.K = K
        self.chart = chart
        self.items = []
        self.ndead = 0
        self.worst = INF
        self.cutoff = INF
    
    def add(self, item):
        entry = [-item.rank_cost(), item]
        heappush(self.items, entry)
        self.chart.index[item] = entry
        self.prune()
   
    def __len__(self):
        return len(self.items) - self.ndead

    def __iter__(self):
        """no order guaranteed"""
        for negcost, item in self.items:
            if not item.dead:
                yield item

    def __getitem__(self, i):
        if not hasattr(self, 'sorted_items'):
            self.sorted_items = [item for negcost, item in self.items]
            self.sorted_items.sort()
        return self.sorted_items[i]

    # ----------- begin of methods class users usually do not need------------

    def prune(self):
        while len(self) > self.K:
            negcost, item = heappop(self.items)
            if item.dead:
                self.ndead -= 1
            else:
                del self.chart.index[item]
                # mark pruned item as dead so the agenda will know
                item.dead = True
                self.chart.pruned += 1
        
        # find value of worst item:
        # pop dead items at the top first
        while self.items[0][1].dead:
            negcost, item = heappop(self.items)
            self.ndead -= 1
        # set worst item and set cutoff value
        self.worst = -self.items[0][0]
        if len(self) == self.K: # filled up
            self.cutoff = self.worst

class Chart(object):
    def __init__(self, fwords, start_symbol):
        self.N = len(fwords)
        self.fwords = fwords
        self.bins = {}
        self.index = {}

        self.start_symbol = start_symbol

        # statistics
        self.pruned = 0
        self.merged = 0
        self.prepruned = 0
        self.unary_cycle_broken = 0
        self.negcost_pruned = 0
        self.neg_unary_pruned = 0

    def add(self, item):
        added = False
        bin_idx = self.key(item)
        if bin_idx:  # discard items with None key
            bin = self.bins.setdefault(bin_idx, Bin(FLAGS.bin_size, self))
            # preprune
            if item.rank_cost() > bin.cutoff:
                if logger.level >= 4:
                    logger.writeln('prepruned: %s' % item)
                self.prepruned += 1
            # TODO: a hack: ban unary negative deduction, 
            # only for ghkm rules
            elif item.incoming[0].rule.arity == 1 and len(item.incoming[0].rule.f) == 1 and \
                 item.incoming[0].cost <= 0 and \
                 item.incoming[0].rule.grammar is not None and \
                 'ghkm' in item.incoming[0].rule.grammar.name:
                if logger.level >= 4:
                    logger.write('negative unary deduction for ghkm banned: %s' % item)
                self.neg_unary_pruned += 1
            # ban negative deduction
            elif FLAGS.ban_negative_deduction and item.incoming[0].cost <= 0:
                if logger.level >= 4:
                    logger.writeln('negative deduction banned: %s' % item.incoming[0])
                self.negcost_pruned += 1
            # unary cycle banned
            elif item.unary_cycle():
                if logger.level >= 4:
                    logger.writeln('unary cycle broken: %s' % item)
                self.unary_cycle_broken += 1
            # merging needed
            elif item in self.index:
                oldcost, olditem = self.index[item]
                item_merged = item.merge(olditem)
                if item_merged:  # old item better
                    if logger.level >= 4:
                        logger.writeln('merged: %s' % item)
                else:  # new item better
                    bin.add(item) 
                    bin.ndead += 1
                    added = True
                self.merged += 1
            # no need to merge
            else:
                bin.add(item)
                added = True
        return added

    def key(self, item):
        """return the index of the bin an item belongs to"""
        if item.goal():
            return 'GOAL'
        elif item.var == FLAGS.glue_var:
            return ('GLUE', item.i, item.j)
        elif is_virtual(item.var):
            return ('VIRTUAL', item.i, item.j)
        elif item.var == FLAGS.straight_var:
            return ('STRAIGHT', item.i, item.j)
        elif item.var == FLAGS.inverted_var:
            return ('INVERTED', item.i, item.j)
        elif item.var == self.start_symbol:
            if item.i != 0:
                return None
            else:
                return ('START', item.j)
        else:
            # TODO: filtering like this doesn't work
            # if item.j - item.i > 10:
            #     return None
            # else:
            return (item.i, item.j)

    def items(self, i, j):
        keys = [(i, j),
                ('GLUE', i, j),
                ('VIRTUAL', i, j),
                ('STRAIGHT', i, j),
                ('INVERTED', i, j)]
        if i == 0:
            keys.append(('START', j))
        for key in keys:
            bin = self.bins.get(key)
            if bin:
                for item in bin:
                    yield item

    def iter_items_by_nts(self, i, j):
        """Yield (nt, bin) pairs."""
        # TODO: this can be made faster by saving the returned result
        # but unary expansion changes a span's items for once
        keys = [(i, j),
                ('GLUE', i, j),
                ('VIRTUAL', i, j),
                ('STRAIGHT', i, j),
                ('INVERTED', i, j)]
        if i == 0:
            keys.append(('START', j))
        items = []
        for key in keys:
            bin = self.bins.get(key)
            if bin:
                for item in bin:
                    items.append(item)
        items.sort()
        var2itemlist = {}
        for item in items:
            nt = item.var
            if FLAGS.nt_mismatch:
                nt = nocat(nt)
            itemlist = var2itemlist.setdefault(nt, [])
            itemlist.append(item)
        for item in var2itemlist.items():
            yield item

    def leftneighbors(self, item):
        for k in range(0, item.i):
            keys = [(k, item.i),
                    ('GLUE', k, item.i),
                    ('VIRTUAL', k, item.i),
                    ('STRAIGHT', k, item.i),
                    ('INVERTED', k, item.i)]
            for key in keys:
                bin = self.bins.get(key)
                if bin:
                    for litem in bin:
                        yield litem
        bin = self.bins.get(('START', item.i))
        if bin:
            for litem in bin:
                yield litem

    def rightneighbors(self, item):
        for k in range(item.j+1, self.N+1):
            keys = [(item.j, k),
                    ('GLUE', item.j, k),
                    ('VIRTUAL', item.j, k),
                    ('STRAIGHT', item.j, k),
                    ('INVERTED', item.j, k)]
            for key in keys:
                bin = self.bins.get(key)
                if bin:
                    for ritem in bin:
                        yield ritem

    def iterbins(self):
        for bin in self.bins.values():
            yield bin

    def __len__(self):
        "number of items in chart"
        result = 0
        for bin in self.iterbins():
            result += len(bin)
        return result

    def stats(self):
        header = '{:-^50}\n'
        field = '{:<35}{:>15}\n'

        nitem = 0
        ndead = 0
        for bin in self.iterbins():
            nitem += len(bin)
            ndead += bin.ndead
            
        result = header.format('Chart Stats')
        result += field.format('[in chart]:', nitem)
        result += field.format('[dead in chart]:', ndead)
        result += field.format('[pruned]:', self.pruned)
        result += field.format('[prepruned]:', self.prepruned)
        result += field.format('[merged]:', self.merged)
        result += field.format('[unary cycle broken]:',
                               self.unary_cycle_broken)
        result += field.format('[negative cost edge pruned]:',
                               self.negcost_pruned)
        result += field.format('[negative unary edge pruned]:',
                               self.neg_unary_pruned)
        return result
