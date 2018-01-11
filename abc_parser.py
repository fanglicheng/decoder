from heapq import heapify, heappop, heappush

from phrase_hypergraph import PhraseHGNode, PhraseHGEdge
from common import bi_cyk_spans
from rule import Rule
from hypergraph import Hypergraph
import logger

logger.level = 1

PHRASE = '[A]'
STRAIGHT = '[STRAIGHT]'
INVERTED = '[INVERTED]'
START = '[S]'

glue_missing_phrases = False

class NeighborIndex(object):
    """ Neighbors:
    0 = upper right
    1 = upper left
    2 = lower left
    3 = lower right
    """
    def __init__(self):
        self.index = {}

    def add(self, node):
        bin = self.index.setdefault((node.fi, node.ei),
                                    [[] for i in range(4)])
        bin[3].append(node)
        bin = self.index.setdefault((node.fj, node.ei),
                                    [[] for i in range(4)])
        bin[0].append(node)
        bin = self.index.setdefault((node.fi, node.ej),
                                    [[] for i in range(4)])
        bin[2].append(node)
        bin = self.index.setdefault((node.fj, node.ej),
                                    [[] for i in range(4)])
        bin[1].append(node)

    def get(self, coordinate, bin_no):
        bin = self.index.get(coordinate)
        if bin is None:
            return []
        else:
            return bin[bin_no]

# not used
class EdgeIndex(object):
    def __init__(self):
        self.index = {}

    def test_and_add(self, edge):
        k = self.key(edge)
        if k in self.index:
            added = False
        else:
            self.index[k] = edge
            added = True
        return added

    def key(self, edge):
        item1, item2 = edge.tail
        return (item1.nt,
                item1.fi,
                item1.fj,
                item1.ei,
                item1.ej,
                item2.nt,
                item2.fi,
                item2.fj,
                item2.ei,
                item2.ej)

class ABCParser(object):
    """Bilingual parser that glues hiero rules into a hypergraph with
    ABC glue grammar."""
    def __init__(self, n1, n2, phrases):
        """
        n1: French length
        n2: English length
        phrases: a list of PhraseHGNodes that have been partially linked
          according to heiro rule extraction
        """
        self.n1 = n1
        self.n2 = n2
        self.chart = {}
        self.neighbor_index = NeighborIndex()
        self.edge_index = set()
        self.agenda = []
        self.phrases = phrases
        self.glue_nodes = []
        for phrase in phrases:
            bin = self.chart.setdefault((phrase.fi,
                                         phrase.fj,
                                         phrase.ei,
                                         phrase.ej),
                                        {})
            bin[phrase.nt] = phrase
            self.agenda.append(phrase)
            self.neighbor_index.add(phrase)

    # not used, too slow
    def parse(self):
        self.glue_nodes = []
        for i1, j1, i2, j2 in bi_cyk_spans(self.n1, self.n2):
            for k1 in range(i1 + 1, j1):
                for k2 in range(i2 + 1, j2):
                    bin1 = self.chart.get((i1, k1, i2, k2), {})
                    bin2 = self.chart.get((k1, j1, k2, j2), {})
                    for item1 in bin1.values():
                        for item2 in bin2.values():
                            if item2.nt != STRAIGHT:
                                new_item = self.make_item(item1,
                                                          item2,
                                                          False)
                                self.chart_add(new_item)
                    bin1 = self.chart.get((i1, k1, k2, j2), {})
                    bin2 = self.chart.get((k1, j1, i2, k2), {})
                    for item1 in bin1.values():
                        for item2 in bin2.values():
                            if item2.nt != INVERTED:
                                new_item = self.make_item(item1,
                                                          item2,
                                                          True)
                                self.chart_add(new_item)
        self.stats()

    def parse_agenda(self):
        while len(self.agenda) > 0:
            item = self.agenda.pop()
            if logger.level >= 4:
                logger.writeln('pop: %s' % item)
            for item1, item2, inverted in self.neighboring_pairs(item):
                # avoid duplicated edges. note that in ABC grammar,
                # if the boxes of item1 and item2 are given, the nt of the
                # new item is fixed
                if logger.level >= 4:
                    logger.writeln('neighbors: %s %s' % (item1, item2))
                key = (item1.nt, item1.fi, item1.fj, item1.ei, item1.ej,
                       item2.nt, item2.fi, item2.fj, item2.ei, item2.ej)
                if key not in self.edge_index:
                    self.edge_index.add(key)  
                    new_item = self.make_item(item1, item2, inverted)
                    if self.chart_add(new_item):
                        self.agenda.append(new_item)
                        self.neighbor_index.add(new_item)
                        self.glue_nodes.append(new_item)
                        if logger.level >= 4:
                            logger.writeln('push: %s' % new_item)
        # self.stats()
        root = self.final_glue()
        self.hg = Hypergraph(root)
        self.hg.topo_sort()
        self.stats()
        return self.hg

    def final_glue(self):
        unattached = self.phrases[:]
        candidates = self.phrases + self.glue_nodes
        # topo sort. root node at the end
        unattached.sort()
        candidates.sort()
        self.top_roots = []
        self.other_roots = []
        while len(candidates) > 0:
            root = candidates.pop()
            if (root.fi == 0 and
                root.fj == self.n1 and
                root.ei == 0 and
                root.ej == self.n2):
                self.top_roots.append(root)
            else:
                self.other_roots.append(root)
            hg = Hypergraph(root)
            hg.find_reachable_nodes()
            unattached = [n for n in unattached if id(n) not in hg.found]
            candidates = [n for n in candidates if id(n) not in hg.found and \
                          (n.nt == PHRASE or not n < root)]
        top_node = PhraseHGNode(START, 0, self.n1, 0, self.n2)
        # add one edge for each top root
        for root in self.top_roots:
            rule = Rule()
            rule.lhs = START
            rule.f = [root.nt]
            rule.e = [root.nt]
            rule.e2f = [0]
            edge = PhraseHGEdge(rule)
            edge.add_tail(root)
            top_node.add_incoming(edge)
        # add one edge for all other roots
        if ((glue_missing_phrases or len(self.top_roots) == 0)
            and len(self.other_roots) > 0):
            rule = Rule()
            rule.lhs = START
            edge = PhraseHGEdge(rule)
            for root in self.other_roots:
                rule.f.append(root.nt)
                rule.e.append(root.nt)
                edge.add_tail(root)
            rule.e2f = [i for i in range(len(rule.f))]
            top_node.add_incoming(edge)
        return top_node

    # not used
    def final_glue1(self):
        """try to cover all phrases AND glue rules"""
        # candidate glue nodes are glue nodes whose boxes are also phrases
        # candidate_glue_nodes = []
        # for node in self.glue_nodes:
        #     bin = self.chart.get((node.fi, node.fj, node.ei, node.ej))
        #     if bin is not None:
        #         if PHRASE in bin:
        #             candidate_glue_nodes.append(node)
        candidates = self.phrases + self.glue_rules
        # topo sort. root node at the end
        candidates.sort()
        roots = []
        while len(candidates) > 0:
            root = candidates.pop()
            print('pop: %s' % root)
            roots.append(root)
            hg = Hypergraph(root)
            hg.find_reachable_nodes()
            candidates = [n for n in candidates if id(n) not in hg.found]
        top_rule = Rule()
        top_rule.lhs = START
        top_edge = PhraseHGEdge(top_rule)
        for root in roots:
            top_rule.f.append(root.nt)
            top_rule.e.append(root.nt)
            top_edge.add_tail(root)
        top_rule.e2f = [i for i in range(len(top_rule.f))]
        top_node = PhraseHGNode(START, 0, self.n1, 0, self.n2)
        top_node.add_incoming(top_edge)
        return top_node

    def neighboring_pairs(self, item):
        """
        return value is items in the order they appear on f side, and whether
        they are inverted.
        The constraint of ABC grammar is also applied here.
        """
        for neighbor in self.neighbor_index.get((item.fi, item.ej), 0):
            if item.nt != INVERTED:
                yield neighbor, item, True
        for neighbor in self.neighbor_index.get((item.fi, item.ei), 1):
            if item.nt != STRAIGHT:
                yield neighbor, item, False
        for neighbor in self.neighbor_index.get((item.fj, item.ei), 2):
            if neighbor.nt != INVERTED:
                yield item, neighbor, True
        for neighbor in self.neighbor_index.get((item.fj, item.ej), 3):
            if neighbor.nt != STRAIGHT:
                yield item, neighbor, False

    def make_item(self, item1, item2, inverted):
        """item1 and item2 is always given in the order they appear
        on the f side"""
        rule = Rule()
        rule.f = [item1.nt, item2.nt]
        fi = item1.fi
        fj = item2.fj
        if inverted:
            rule.lhs = INVERTED
            rule.e = [item2.nt, item1.nt]
            rule.e2f = [1, 0]
            ei = item2.ei
            ej = item1.ej
        else:
            rule.lhs = STRAIGHT
            rule.e = [item1.nt, item2.nt]
            rule.e2f = [0, 1]
            ei = item1.ei
            ej = item2.ej
        edge = PhraseHGEdge(rule)
        edge.add_tail(item1)
        edge.add_tail(item2)
        new_item = PhraseHGNode(rule.lhs, fi, fj, ei, ej)
        new_item.add_incoming(edge)
        return new_item

    def chart_add(self, item):
        bin = self.chart.setdefault((item.fi,
                                     item.fj,
                                     item.ei,
                                     item.ej),
                                    {})
        added = False
        # the ABCParser applies only glue rules. this test says glue rules
        # are used only when a PHRASE is not already derived for the box
        # if PHRASE not in bin:
        old_item = bin.get(item.nt)
        if old_item:
            old_item.add_incoming(item.incoming[0])
        else:
            added = True
            bin[item.nt] = item
        return added


    def stats(self):
        result = '--ABCParser Stats--\n'

        top_bin = self.chart.get((0, self.n1, 0, self.n2))
        if top_bin is None:
            result += 'parse failed\n'
        else:
            result += 'parse succeeded\n'

        result += self.hg.stats()

        # self.hg.show()

        hiero_rules = 0
        glue_rules = []
        for edge in self.hg.edges():
            if edge.rule.lhs == PHRASE:
                hiero_rules += 1
            else:
                glue_rules.append(edge)
        result += 'hiero rules: %s\n' % hiero_rules
        result += 'glue rules: %s\n' % len(glue_rules)

        rules = []
        for node in self.phrases:
            for edge in node.incoming:
                rules.append(edge.rule)
        hg_rules = set()
        for edge in self.hg.edges():
            hg_rules.add(id(edge.rule))
        unglued_rules = []
        for rule in rules:
            if id(rule) not in hg_rules:
                unglued_rules.append(rule)

        roots = self.top_roots + self.other_roots
        result += 'roots: %s\n' % len(roots)
        for node in roots:
            result += '%s\n' % node

        result += 'unglued rules: %s\n' % len(unglued_rules)
        for rule in unglued_rules:
            result += '%s\n' % rule
        return result
