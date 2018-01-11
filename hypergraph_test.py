#!/usr/bin/env python3
from hypergraph import Node, Edge, Hypergraph, INSIDE, LOGPROB
import unittest
from unittest import TestCase
from math import log

import logger

class ForestNode(Node):
    def __init__(self, label):
        Node.__init__(self)
        self.label = label

    def __str__(self):
        result = self.label
        if hasattr(self, 'inside'):
            result += r'\nI: %s' % self.inside
        if hasattr(self, 'outside'):
            result += r'\nO: %s' % self.outside
        if hasattr(self, 'inside') and hasattr(self, 'outside'):
            result += r'\nI*O: %s' % self.hg.prod(self.inside, self.outside)
        if hasattr(self, 'inside_exp'):
            result += r'\nIexp: %s' % self.inside_exp
        if hasattr(self, 'outside_exp'):
            result += r'\nOexp: %s' % self.outside_exp
        if hasattr(self, 'inside_exp') and hasattr(self, 'outside_exp'):
            result += r'\nIexp*O+Oexp*I: %s' \
                    % self.hg.sum(self.hg.prod(self.inside_exp, self.outside), 
                                  self.hg.prod(self.outside_exp, self.inside))
        return result

class ForestEdge(Edge):
    def __init__(self):
        Edge.__init__(self)
    
    def __str__(self):
        result = str(self.hg.w(self))
        if 'inside' in self.hg.tasks_done and 'outside' in self.hg.tasks_done:
            result += r'\nP(e): %s' % self.posterior()
        if 'inside_exp' in self.hg.tasks_done and \
           'outside_exp' in self.hg.tasks_done:
            result += r'\nE(e): %s' % self.expectation()
        return result

class InsideOutsideTest(TestCase):
    def setUp(self):
        w0 = ForestNode('John')
        w1 = ForestNode('saw')
        w2 = ForestNode('a')
        w3 = ForestNode('girl')
        w4 = ForestNode('with')
        w5 = ForestNode('a')
        w6 = ForestNode('telescope')
        t0_1 = ForestNode('NN')
        t1_2_0 = ForestNode('VB')
        t1_2_1 = ForestNode('NN')
        t2_3 = ForestNode('DT')
        t3_4 = ForestNode('NN')
        t4_5 = ForestNode('IN')
        t5_6 = ForestNode('DT')
        t6_7 = ForestNode('NN')
        t2_4 = ForestNode('NP')
        t5_7 = ForestNode('NP')
        t1_4 = ForestNode('VP')
        t4_7 = ForestNode('PP')
        t2_7 = ForestNode('NP')
        t1_7 = ForestNode('VP')
        root = ForestNode('S')
        # [NN,0,1] -> John
        e = ForestEdge()
        e.add_tail(w0)
        e.prob = 0.02
        t0_1.add_incoming(e)
        # [VB,1,2] -> saw
        e = ForestEdge()
        e.add_tail(w1)
        e.prob = 0.01
        t1_2_0.add_incoming(e)
        # [NN,1,2] -> saw
        e = ForestEdge()
        e.add_tail(w1)
        e.prob = 0.01
        t1_2_1.add_incoming(e)
        # [DT,2,3] -> a
        e = ForestEdge()
        e.add_tail(w2)
        e.prob = 0.5
        t2_3.add_incoming(e)
        # [NN,3,4] -> girl
        e = ForestEdge()
        e.add_tail(w3)
        e.prob = 0.05
        t3_4.add_incoming(e)
        # [IN,4,5] -> with
        e = ForestEdge()
        e.add_tail(w4)
        e.prob = 0.25
        t4_5.add_incoming(e)
        # [DT,5,6] -> a
        e = ForestEdge()
        e.add_tail(w5)
        e.prob = 0.5
        t5_6.add_incoming(e)
        # [NN,6,7] -> telescope
        e = ForestEdge()
        e.add_tail(w6)
        e.prob = 0.001
        t6_7.add_incoming(e)
        # [NP,2,4] -> [DT,2,3] [NN,3,4]
        e = ForestEdge()
        e.add_tail(t2_3)
        e.add_tail(t3_4)
        e.prob = 0.7
        t2_4.add_incoming(e)
        # [NP,5,7] -> [DT,5,6] [NN,6,7]
        e = ForestEdge()
        e.add_tail(t5_6)
        e.add_tail(t6_7)
        e.prob = 0.7
        t5_7.add_incoming(e)
        # [VP,1,4] -> [VB,1,2] [NP,2,4]
        e = ForestEdge()
        e.add_tail(t1_2_0)
        e.add_tail(t2_4)
        e.prob = 0.9
        t1_4.add_incoming(e)
        # [PP,4,7] -> [IN,4,5] [NP,5,7]
        e = ForestEdge()
        e.add_tail(t4_5)
        e.add_tail(t5_7)
        e.prob = 1.0
        t4_7.add_incoming(e)
        # [NP,2,7] -> [NP,2,4] [PP,4,7]
        e = ForestEdge()
        e.add_tail(t2_4)
        e.add_tail(t4_7)
        e.prob = 0.3
        t2_7.add_incoming(e)
        # [VP,1,7] -> [VB,1,2] [NP,2,7]
        e = ForestEdge()
        e.add_tail(t1_2_0)
        e.add_tail(t2_7)
        e.prob = 0.5
        t1_7.add_incoming(e)
        # [VP,1,7] -> [VP,1,4] [PP,4,7]
        e = ForestEdge()
        e.add_tail(t1_4)
        e.add_tail(t4_7)
        e.prob = 0.5
        t1_7.add_incoming(e)
        # [S,0,7] -> [NN,0,1] [VP,1,7]
        e = ForestEdge()
        e.add_tail(t0_1)
        e.add_tail(t1_7)
        e.prob = 0.9
        root.add_incoming(e)

        self.hp = Hypergraph(root)

    def test_inside_outside(self):
        self.hp.set_semiring(INSIDE)
        self.hp.set_functions(lambda x: x.prob, lambda x: 1, None)
        self.hp.inside()
        self.hp.outside()
        logger.writeln(self.hp.dot())
        # self.hp.show()

    def test_inside_exp_outside_exp(self):
        self.hp.set_semiring(INSIDE)
        self.hp.set_functions(lambda x: x.prob, lambda x: 1, None)
        self.hp.inside()
        self.hp.outside()
        self.hp.inside_exp()
        self.hp.outside_exp()
        logger.writeln(self.hp.dot())
        # self.hp.show()

    def test_inside_outside_log(self):
        self.hp.set_semiring(LOGPROB)
        self.hp.set_functions(lambda x: log(x.prob), lambda x: 1, None)
        self.hp.inside()
        self.hp.outside()
        logger.writeln(self.hp.dot())
        # self.hp.show()

    def test_inside_exp_outside_exp_log(self):
        self.hp.set_semiring(LOGPROB)
        self.hp.set_functions(lambda x: log(x.prob), lambda x: 1, None)
        self.hp.inside()
        self.hp.outside()
        self.hp.inside_exp()
        self.hp.outside_exp()
        logger.writeln(self.hp.dot())
        # self.hp.show()

    def test_best_paths(self):
        self.hp.set_semiring(INSIDE)
        self.hp.set_functions(lambda x: x.prob, lambda x: 1, None)
        self.hp.assert_done('topo_sort')
        logger.writeln(self.hp.root.best_paths()[0].tree_str())
        logger.writeln(self.hp.root.best_paths()[0].weight)
        logger.writeln(self.hp.root.best_paths()[1].tree_str())
        logger.writeln(self.hp.root.best_paths()[1].weight)

if __name__ == '__main__':
    unittest.main()
