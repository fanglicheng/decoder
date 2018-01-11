#!/usr/bin/env python3

import sys
import time
import os

from hypergraph import Node, Edge
from consensus_training import cartesian
import rule_filter

class PhraseHGNode(Node):
    def __init__(self, nt, fi, fj, ei, ej):
        """(i1, j1, i2, j2) is a box in the alignment matrix.
        nt is the nonterminal for this box."""
        Node.__init__(self)
        self.nt = nt
        self.fi = fi
        self.fj = fj
        self.ei = ei
        self.ej = ej

    def __str__(self):
        if hasattr(self, 'id'):
            result = '%s [%s,%s,%s,%s,%s]' % (self.id,
                                              self.nt,
                                              self.fi,
                                              self.fj,
                                              self.ei,
                                              self.ej)
        else:
            result = '[%s,%s,%s,%s,%s]' % (self.nt,
                                           self.fi,
                                           self.fj,
                                           self.ei,
                                           self.ej)
        return result

    def __lt__(self, other):
        if ((other.fi <= self.fi and
             self.fj <= other.fj and
             other.ei <= self.ei and
             self.ej <= other.ej) and
            (not (other.fi == self.fi and
                  self.fj == other.fj and
                  other.ei == self.ei and
                  self.ej == other.ej))):
            return True
        else:
            return False

    def make_composed_rules(self):
        self.rules = []
        if self.fj - self.fi > 10 or self.ej - self.ei > 10:
            return
        for edge in self.incoming:
            for subrules in cartesian([[None] + tailnode.rules
                                      for tailnode in edge.tail]):
                rule = edge.rule.compose(subrules)
                rule.level = sum(sub.level for sub in subrules
                                 if sub is not None) + 1
                if rule_filter.filter(rule):
                    self.rules.append(rule)

class PhraseHGEdge(Edge):
    def __init__(self, rule=None):
        Edge.__init__(self)
        self.rule = rule

    def __str__(self):
        return '%s' % self.rule

    def serialize(self):
        edge_str = Edge.serialize(self)
        rule_str = str(self.rule)
        return ' ||||| '.join([edge_str, rule_str])

    def deserialize(self):
        edge_str, rule_str = line.split('|||||')
        tail_ids, head_id = Edge.deserialize(self, edge_str)
        self.rule = Rule()
        self.rule.fromstr(rule_str)
        return tail_ids, head_id
