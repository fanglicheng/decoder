#!/usr/bin/env python
#!/usr/grads/lib/pypy-pypy-64bit/pypy/translator/goal/pypy-c

from __future__ import print_function
from __future__ import division

import sys
import time
import pickle
import os
import random
from math import log, exp, factorial, lgamma

#import syspath
import logprob
import alignment
from phrase_forest import make_rule, phrase_decomposition_forest
import logger
import gflags
from lexical_weighter import LexicalWeighter
from common import INF, ZERO
from levels import mark_level
from monitor import memory, resident

FLAGS = gflags.FLAGS

PHRASE_NT = 'X'

gflags.DEFINE_string(
    'base',
    'poisson',
    'Base distribution')
gflags.DEFINE_float(
    'alpha',
    5.0,
    'Concentration parameter in Dirichlet process.')
gflags.DEFINE_float(
    'discount',
    0.5,
    'Discount parameter in Pitman-yor process.')
gflags.DEFINE_integer(
    'maxcut',
    7,
    'no cut sampling when target or source span larger than maxcut.')
gflags.DEFINE_boolean(
    'sample_cut_only',
    False,
    'Sample at only nodes already cut (wrong implementation).')
gflags.DEFINE_integer(
    'sample_level',
    None,
    'Sample only nodes with level <= sample_level."')
gflags.DEFINE_integer(
    'level_inc',
    None,
    'Sample each level for #level_inc# iterations."')
gflags.DEFINE_boolean(
    'double_count',
    False,
    'Use double counting."')
gflags.DEFINE_boolean(
    'variable_alpha',
    False,
    'concentration parameter is different for each length')
gflags.DEFINE_boolean(
    'correct_edge_sampling',
    False,
    '"correct" path sampling. Number of incoming nodes is considered')
gflags.DEFINE_boolean(
    'lhs_conditional',
    False,
    'normalized over lhs.')
gflags.DEFINE_boolean(
    'seed_random',
    False,
    'seed random number generation.')
gflags.DEFINE_boolean(
    'sample_cut',
    True,
    'sample cut point (set to no to sample only minimal rules)')
gflags.DEFINE_boolean(
    'sample_edge',
    True,
    'sample edge switch')
gflags.DEFINE_integer(
    'splits',
    2,
    'two-way nt split by default, but you can change it.')
gflags.DEFINE_string(
    'model',
    'PY',
    'model. choose from DP|PY')
gflags.DEFINE_integer(
    'split_iter',
    10,
    'do symbol split every #split_iter iterations')
gflags.DEFINE_boolean(
    'split_at_iter_0',
    True,
    'allow split to happen at iter 0')
gflags.DEFINE_boolean(
    'type',
    False,
    'use type-based sampling')
gflags.DEFINE_boolean(
    'refine',
    False,
    'use symbol refinement')
gflags.DEFINE_boolean(
    'check_index',
    False,
    'Check cut index for errors.')
gflags.DEFINE_boolean(
    'random_choice',
    False,
    'Use random choice for type-based sampling for debugging.')

def timed(l):
    prev = time.time()
    for i, x in enumerate(l, 1):
        if i % FLAGS.interval == 0:
            logger.writeln('%s (%s/sec)' %
                           (i, FLAGS.interval/(time.time()-prev)))
            prev = time.time()
        yield x

def lncr(n, r):
    "log choose r from n"
    return lgamma(n+1) - lgamma(r+1) - lgamma(n-r+1)

def discrete(l):
    s = sum(l)
    if s == 0.0:
        length =  len(l)
        l = [1/length] * length
    else:
        l = [x/s for x in l]
    s = 0.0
    r = random.random()
    for i, x in enumerate(l):
        s += x
        if r <= s:
            return i

def rule_size(rule):
    return  len(rule.f) + len(rule.e) - 2 * rule.arity + rule.scope()

def geometric(rule):
    p = 0.99

    l = rule_size(rule)
    if l == 0:
        return 1.0
    else:
        return (1-p)**(l-1) * p

def poisson(rule):
    mean = 2.0
    l = rule_size(rule)

    denominator = log(factorial(l))
    numerator = l * log(mean) - mean

    result = exp(numerator - denominator)

    return result if result > ZERO else ZERO

def weibull(rule):
    shape = 2.9
    scale = 2.3

    l = rule_size(rule)

    result = ((shape/scale)
            * (pow(float(l)/scale, shape-1))
            * exp(-pow(float(l)/scale, shape)))

    return result if result > ZERO else ZERO

def uniform(rule):
    return 1.0

# emulate a Dirichlet distribution with three events
# used for the ab_test
def abtest_base(base):
    return 1.0/3

#def uniform_base(rule):
#    vocab_size_e = 34245 + 1
#    vocab_size_f = 32492 + 1
#    e_prob = pow(vocab_size_f, -float(len(rule.f)))
#    f_prob = pow(vocab_size_e, -float(len(rule.e)))
#    
#    result = e_prob * f_prob
#
#    return result if result > ZERO else ZERO

def uniform_base(rule):
    l = rule_size(rule)

    result = pow(2.0, -l)
    return result if result > ZERO else ZERO

base = poisson
rule_size_prob = poisson

def choose_m(n, c1, c2, sampler):
    p = 0.0
    weights = [0] * (n+1)
    for i in xrange(n):
        for x in c1:
            p += logprob.elog(sampler.posterior(x))
            sampler.count(x)
    weights[n] = p
    for i in xrange(n):
        for x in c1:
            sampler.discount(x)
            p -= logprob.elog(sampler.posterior(x))
        for x in c2:
            p += logprob.elog(sampler.posterior(x))
            sampler.count(x)
        weights[n-i-1] = p
    # discount all c2's
    for i in xrange(n):
        for x in c2:
            sampler.discount(x)
    for i in xrange(n+1):
        weights[i] += lncr(n, i)
    weights = [logprob.eexp(w) for w in weights]
    #print(weights)
    return discrete(weights)

class Sampler(object):
    def count(self, x):
        pass

    def discount(self, x):
        pass

    def posterior(self, x):
        return 1.0

    def choice_posterior(self, c):
        result = 1.0
        for x in c:
            result *= self.posterior(x)
            if result < ZERO:
                return ZERO
        return result

class NPSampler(Sampler):
    "NP for non-paramatric"
    def __init__(self):
        self.counts = {}
        self.rule_size_counts = {}  # key=rule size, value=counts
        self.rule_size_tables = {}  # key=rule size, value=estimated tables
        self.n = 0

    def update_rule_size_tables(self):
        self.rule_size_tables = {}

        # should be fixed for python3
        for rule, counts in self.counts.iteritems():
            l = rule_size(rule)
            estimated_number_of_tables = pow(counts, FLAGS.discount)
            self.rule_size_tables[l] =  (self.rule_size_tables.get(l, 0.0)
                                            + estimated_number_of_tables)

#    def pitman_yor_posterior(self, c):
#        result = 1.0
#        for x in c:
#            l = rule_size(x)
#            n_r = self.counts.get(x, 0.0)
#            T_r = pow(n_r, FLAGS.discount)
#            T_e = pow(self.n, FLAGS.discount)
#
#            result *= ((n_r - T_r*FLAGS.discount + (T_e + FLAGS.alpha)*base(x))
#                        / (self.n + FLAGS.alpha))
#
#            if result < ZERO:
#                return ZERO
#
#        return result

    def pitman_yor_posterior_rule_size(self, x):
        alpha = FLAGS.alpha
        if FLAGS.variable_alpha == True:
            alpha = FLAGS.alpha * rule_size_prob(x)

        l = rule_size(x)
        n_r = self.counts.get(x, 0.0)
        n_l = self.rule_size_counts.get(l, 0.0)
        T_r = pow(n_r, FLAGS.discount)
        T_l = self.rule_size_tables.get(l, 0.0)

        return (((n_r - T_r*FLAGS.discount + (T_l*FLAGS.discount + alpha)*base(x))
                * (rule_size_prob(x)))
                / (n_l + alpha))

    # Used for generating likelihood graph
    # Single Dirichlet process, no rule size
    def simple_dirichlet_posterior(self, x):
        n_r = self.counts.get(x, 0.0)
        return ((n_r + FLAGS.alpha*base(x))
                / (self.n + FLAGS.alpha))

    def simple_dirichlet_posterior_for_choice(self, c):
        result = 1.0

        alpha = FLAGS.alpha
        for x in c:
            alpha = FLAGS.alpha
            n_r = self.counts.get(x, 0.0)

            result *= ((n_r + alpha*base(x))
                        / (self.n + alpha))

            self.count(x)

        # To make it absolutely correct
        for x in c:
            self.discount(x)

        return result

    def count(self, x):
        #print('count %s' % x)
        self.counts[x] = self.counts.get(x, 0) + 1
        self.n += 1

        if FLAGS.model == 'PY':
            l = rule_size(x)
            self.rule_size_counts[l] = self.rule_size_counts.get(l, 0) + 1

    def discount(self, x):
        #print('discount %s' % x)
        if FLAGS.model == 'PY':
            l = rule_size(x)
            try:
                c = self.rule_size_counts[l]
                if c == 1:
                    del self.rule_size_counts[l]
                else:
                    self.rule_size_counts[l] = c - 1
            except KeyError:
                print('Warning: rule size %d not seen before in discounting' % l,
                      file=sys.stderr)

        try:
            c = self.counts[x]
            if c == 1:
                del self.counts[x]
            else:
                self.counts[x] = c - 1
            self.n -= 1
        except KeyError:
            print('Warning: %s not seen before in discounting' % x,
                  file=sys.stderr)

    def double_count(self, node):
        rule = make_composed_rule(node)
        #print('count %s' % rule)
        c = self.counts.get(rule, 0)
        if c == 0:
            for child in node.incoming[node.edge].tail:
                if not child.cut:
                    self.double_count(child)
        self.counts[rule] = c + 1
        self.n += 1
        l = rule_size(rule)
        self.rule_size_counts[l] = self.rule_size_counts.get(l, 0) + 1

    def double_discount(self, node):
        rule = make_composed_rule(node)
        l = rule_size(x)
        #print('discount %s' % rule)
        c = self.counts.get(rule)
        assert c is not None, '%s not seen before in discounting' % rule
        c -= 1
        if c == 0:
            del self.counts[rule]
            del self.rule_size_counts[l]
            for child in node.incoming[node.edge].tail:
                if not child.cut:
                    self.double_discount(child)
        else:
            self.counts[rule] = c
        self.n -= 1
        self.rule_size_counts[l] -= 1

    # Now this is really wrong...
    def rule_size_likelihood(self):
        result = 0.0
        for rule, count in self.counts.iteritems():
            l = rule_size(rule)
            f_n_l = float(self.rule_size_counts[l])
            l_prob = rule_size_prob(rule)
            result += count * logprob.elog(float(count) * l_prob / f_n_l)
        return result

    # Only works for dirichlet process, no rule size
    def dp_likelihood(self):
        n = 0.0
        counts = {}
        result = 0.0
        alpha = FLAGS.alpha
        for r, count in self.counts.iteritems():
            for _ in range(count):
                n_r = float(counts.get(r, 0.0))
                result += logprob.elog((n_r+alpha*base(r))/(n+alpha))
                counts[r] = counts.get(r, 0.0) + 1.0 
                n += 1.0
        return result

    def nsamples(self):
        return self.n

    def ntypes(self):
        return len(self.counts)

class NTSampler(Sampler):
    "NT for nonterminal"
    def __init__(self):
        self.samplers = {}

    def count(self, x):
        self.samplers.setdefault(x.lhs, NPSampler()).count(x)

    def discount(self, x):
        self.samplers.setdefault(x.lhs, NPSampler()).discount(x)

    def posterior(self, x):
        return self.samplers.setdefault(x.lhs, NPSampler()).posterior(x)

    def update_rule_size_tables(self):
        for sampler in self.samplers.itervalues():
            sampler.update_rule_size_tables()

    def nsamples(self):
        return sum(s.nsamples() for s in self.samplers.itervalues())

    def ntypes(self):
        return sum(s.ntypes() for s in self.samplers.itervalues())

    def likelihood(self):
        return sum(s.likelihood() for s in self.samplers.itervalues())

def init_split(samples, split=True):
    global SAMPLER
    logger.writeln('initialization. split=%s' % split)
    SAMPLER = init_sampler()
    for sample in timed(samples):
        if split:
            for node in sample.hg.nodes:
                node.pnt = node.nt
                node.nt = random.choice(child_symbols(node.pnt))
        for n, rule in sample.composed_rules_under(sample.hg.root):
            SAMPLER.count(rule)
            if FLAGS.type:
                # mapping from rules to nodes, and from nodes to rules
                CUT_INDEX.add(rule, sample, n)
                n.composed_rule = rule
        if FLAGS.check_index:
            CUT_INDEX.check(sample)

def child_symbols(nt):
    "Return range of symbol indices given parent symbol index"
    return range(nt*FLAGS.splits+1, (nt+1)*FLAGS.splits+1)

def parent_symbol(nt):
    return (nt - 1)//2

def choice(choices, i, correction=None):
    "Uses a global sampler to choose among a few options."
    # when we use double count sample_cut and sample_edge
    # call Sampler.double_count and Sampler.double_discount
    # by themselves
    if not FLAGS.double_count:
        for x in choices[i]:
            SAMPLER.discount(x)

    if FLAGS.correct_edge_sampling or correction is None:
        result = discrete([SAMPLER.choice_posterior(c)
                           for c in choices])
    else:
        assert len(choices) == len(correction), \
               'len(choices)=%s, len(correction)=%s' % (len(choices),
                                                        len(correction))
        result = discrete([SAMPLER.choice_posterior(c)*f
                          for c, f in zip(choices, correction)])

    if not FLAGS.double_count:
        for x in choices[result]:
            SAMPLER.count(x)

    return result

def children(node):
    edge = node.incoming[node.edge]
    result = []
    for n in edge.tail:
        if n.cut:
            result.append(n)
        else:
            result.extend(children(n))
    return result

def cut_nodes_under(node):
    "pre-order, self not included"
    for child in children(node):
        yield child
        for c in cut_nodes_under(child):
            yield c

def nodes_turned_on_under(node):
    result = []
    queue = [node]
    while len(queue) > 0:
        curr = queue.pop(0)
        result.append(curr)
        for child in curr.incoming[curr.edge].tail:
            queue.append(child)
    return result

def nodes_in_fragment_under(node):
    "return all nodes that are in the fragment marked by cut points, including self, pre-order"
    yield node
    if not node.cut:
        for n in node.incoming[node.edge].tail:
            for n1 in nodes_in_fragment_under(n):
                yield n1

def conflict_nodes(node, parent):
    "a list of nodes whose cut type depends on this node"
    old = node.cut
    assert parent.cut
    parent.cut = 0
    node.cut = 0
    nodes = list(nodes_in_fragment_under(parent))
    node.cut = old
    parent.cut = 1
    return nodes

class ConflictTester(object):
    def __init__(self):
        self.nodes = set()

    def ok(self, node):
        #print('test')
        #print('existing')
        #for n in self.nodes:
        #    print (id(n))
        #    print (n)
        zone = conflict_nodes(node, cut_parent(node))
        #print('zone')
        for n in zone:
        #    print(id(n))
        #    print(n)
            if n in self.nodes:
                #print ('not ok')
                return False
        self.nodes |= set(zone)
        #print ('ok')
        return True

def density_factor(node):
    "How many choices are there under this node?"
    selected_nodes = nodes_turned_on_under(node)
    #for node in selected_nodes:
    #    print(node, len(node.incoming))
    result = 1
    for n in selected_nodes:
        result *= len(n.incoming)
    result *= 2**len(selected_nodes)
    #print(result)
    return result

def get_nt(node):
    if FLAGS.refine:
        return '[%s-%s]' % (PHRASE_NT, node.nt)
    else:
        return '[%s]' % PHRASE_NT

def init_sampler():
    if FLAGS.lhs_conditional:
        sampler = NTSampler()
    else:
        sampler = NPSampler()

    if FLAGS.model == 'PY':
        NPSampler.likelihood = NPSampler.rule_size_likelihood
        NPSampler.posterior = NPSampler.pitman_yor_posterior_rule_size
    elif FLAGS.model == 'DP':
        NPSampler.likelihood = NPSampler.dp_likelihood
        NPSampler.posterior = NPSampler.simple_dirichlet_posterior
    else:
        assert False, 'unsupported model'
    return sampler

class TreeFile():
    def __init__(self, filename):
        self.f = open(filename, 'w')
        self.i = 1

    def dump(self, sample):
        self.f.write('# %s\n' % self.i)
        self.f.write('%s\n' % sample.tree_str())
        self.i += 1

    def close(self):
        self.f.close()

def dump_trees(samples, filename):
    logger.writeln('dump trees')
    treefile = TreeFile(filename)
    for s in timed(samples):
        # call this before dumping rules for each sample!
        LEXICAL_WEIGHTER.compute_lexical_weights(s.a)
        treefile.dump(s)
    treefile.close()

def choose_k(n):
    return random.randrange(n+1)

def cut_parent(node):
    #print(node)
    assert hasattr(node, 'parent')
    p = node.parent
    while not p.cut:
        p = p.parent
    return p

def set_parents_under(node):
    for child in node.incoming[node.edge].tail:
        child.parent = node
        set_parents_under(child)

def index(node):
    p = node.parent
    n = node
    result = []
    while True:
        for i, c in enumerate(p.incoming[p.edge].tail):
            if c is n:
                result.append(i)
        if p.cut:
            break
        n = p
        p = p.parent
    result.reverse()
    return result

def verify_site(sample, node, r1, r2, r3):
    #print_site(sample, node, r1, r2, r3)

    parent = cut_parent(node)
    old = node.cut
    node.cut = 0
    assert sample.make_composed_rule(parent) == r1, '%s %s' % (sample.make_composed_rule(parent), r1)
    node.cut = 1
    assert sample.make_composed_rule(parent) == r2
    assert sample.make_composed_rule(node) == r3
    node.cut = old

def check_site(sample, node, r1, r2, r3):
    parent = cut_parent(node)
    old = node.cut
    node.cut = 0
    # Do not use != to do comparison here. Python 3 does not do
    # this operator inferrence?
    if not sample.make_composed_rule(parent) == r1:
        node.cut = old
        return False
    node.cut = 1
    if not sample.make_composed_rule(parent) == r2:
        node.cut = old
        return False
    if not sample.make_composed_rule(node) == r3:
        node.cut = old
        return False
    node.cut = old
    return True

def print_site(sample, node, r1, r2, r3):
    print('sample id: %s' % id(sample))
    print('node: %s' % node)
    print('composed rule under node: %s' % sample.make_composed_rule(node))
    print('r1: %s' % r1)
    print('r2: %s' % r2)
    print('r3: %s' % r3)
    print(sample)

class CutType(object):
    def __init__(self):
        pass

RULE_POOL = {}

class CutTypeIndex(object):
    def __init__(self, samples):
        # mapping from rule to nodes
        self.index = {}
        self.samples = samples

    def get(self, rule):
        #print('get %s' % rule)
        sites = self.index.get(rule)
        if sites is None:
            return []
        else:
            return sites

    def add(self, rule, sample, node):
        #print('add %s' % rule)
        self.index.setdefault(rule, set()).add((sample, node))

    def remove(self, rule, sample, node):
        #print('remove %s' % rule)
        #print('key:')
        #print(sample)
        #print('at node %s %s' % (node, sample.make_composed_rule(node)))
        nodeset = self.index.get(rule)
        #print('in node set')
        #for s, n in nodeset:
        #    if (s is sample):
        #        print(s)
        #        print(n)
        #        print(s.make_composed_rule(n))
        try:
            nodeset.remove((sample, node))
        except:
            print(rule)
            print(sample)
            print(node)
            raise
        if len(nodeset) == 0:
            del self.index[rule]

    def test(self, rule, sample, node):
        "return if a certain node generates a certain rule."
        nodeset = self.index.get(rule)
        if nodeset:
            return (sample, node) in nodeset
        else:
            return False

    def check(self, sample):
        node2rule = {}
        for n, r in sample.composed_rules_under(sample.hg.root):
            assert n not in node2rule, \
                    '%s\n%s yielded twice by composed_rule_under' % n
            node2rule[n] = r
        node2rule_index = {}
        for r, nodeset in self.index.iteritems():
            for s, n in nodeset:
                if s is sample:
                    assert n not in node2rule_index, \
                            '%s\n%s is duplicated in index' % (s, n)
                    node2rule_index[n] = r
                    assert r == node2rule[n], \
                            '%s\n%s\nrule in index: %s actual rule: %s' % (s, n, r, node2rule[n])
        #logger.writeln('Index check OK')


# not used
def get_cut_sites(sample, node, parent):
    old = node.cut
    node.cut = 0
    # r1 = r2 + r3
    r1 = sample.make_composed_rule(parent)
    node.cut = 1
    r2 = sample.make_composed_rule(parent)
    r3 = sample.make_composed_rule(node)
    node.cut = old

    # intersection of r2 and r3 sites
    r2r3_sites = []
    # TODO: site conflict handled naively here
    for s, n in CUT_INDEX.get(r3):
        if s is not sample and s.make_composed_rule(cut_parent(n)) == r2:
            r2r3_sites.append((s, n))
    r1_sites = []
    for s, n in CUT_INDEX.get(r1):
        if s is not sample:
            r1_sites.append((s, n))
    return r2r2_sites, r1_sites

class CutSite(object):
    def __init__(self, sample, node, parent):
        assert parent is not None
        assert node is not parent
        self.sample = sample
        self.node = node
        self.parent = parent
        # tree fragments: t1 = t2 + t3
        old = node.cut
        node.cut = 0
        t1 = sample.make_composed_rule(parent)
        self.t1 = RULE_POOL.setdefault(t1, t1)
        node.cut = 1
        t2 = sample.make_composed_rule(parent)
        self.t2 = RULE_POOL.setdefault(t2, t2)
        t3 = sample.make_composed_rule(node)
        self.t3 = RULE_POOL.setdefault(t3, t3)
        node.cut = old
        self.type = (t1, t2, t3)

    def update(self, parent):
        sites = CUT_TYPES.setdefault(self.type, set())

class Sample():
    def __init__(self, hg, a):
        self.hg = hg  # hypergraph
        self.a = a  # alignment

        mark_level(self.hg)
        for node in self.hg.nodes:
            node.cut = 1
            #if node.fj - node.fi <= FLAGS.sample_max_phrase_len:
            #    if random.random() <= 0.9:
            #        node.cut = True
            #    else:
            #        node.cut = False
            node.edge = random.randrange(len(node.incoming))
            # this is wrong. there may be multiple parents trying to claim a child.
            #for c in node.incoming[node.edge].tail:
            #    c.parent = node
            node.nt = 0
            node.pnt = None  # parent nt (before splitting)
            # sampled in a particular iteration
            node.sampled = False

        set_parents_under(self.hg.root)

    def make_one_level_rule(self, node):
        mychildren = node.incoming[node.edge].tail
        rule = make_rule([node.fi, node.fj, node.ei, node.ej],
                         [[c.fi, c.fj, c.ei, c.ej] for c in mychildren],
                         self.a.fwords,
                         self.a.ewords,
                         get_nt(node),
                         [get_nt(c) for c in mychildren])
        return rule

    def make_composed_rule(self, node):
        mychildren = children(node)
        rule = make_rule([node.fi, node.fj, node.ei, node.ej],
                         [[c.fi, c.fj, c.ei, c.ej] for c in mychildren],
                         self.a.fwords,
                         self.a.ewords,
                         get_nt(node),
                         [get_nt(c) for c in mychildren])
        return rule

    def composed_rules_under(self, node):
        mychildren = children(node)
        rule = make_rule([node.fi, node.fj, node.ei, node.ej],
                         [[c.fi, c.fj, c.ei, c.ej] for c in mychildren],
                         self.a.fwords,
                         self.a.ewords,
                         get_nt(node),
                         [get_nt(c) for c in mychildren])
        yield node, rule
        for child in mychildren:
            for n, rule in self.composed_rules_under(child):
                yield n, rule

    def cut_type(self, node, parent):
        old = node.cut
        node.cut = 0
        rule = self.make_composed_rule(parent)
        node.cut = old
        return rule

    def tree_str(self):
        return self.tree_str_helper(self.hg.root)

    def tree_str_helper(self, node, indent=0):
        result = ''
        rule = self.make_composed_rule(node)
        rule.feats = [1.0]
        rule.feats.extend(LEXICAL_WEIGHTER.score_rule(self.a, rule))
        result += ' '*indent + str(rule) + '\n'
        for child in children(node):
            result += self.tree_str_helper(child, indent + 4)
        return result

    def sample_edge(self, node, parent):
        if parent is None:  # root node
            parent = node
        if len(node.incoming) == 1:  # trivial case
            return
        rule_lists = []
        old = node.edge

        if FLAGS.double_count:
            SAMPLER.double_discount(parent)
            for cut_node in cut_nodes_under(parent):
                SAMPLER.double_discount(cut_node)

        density_factors = []
        for i in range(len(node.incoming)):
            node.edge = i
            rule_lists.append([r for n, r in self.composed_rules_under(parent)])
            density_factors.append(density_factor(node))

        if FLAGS.random_choice:
            i = random.randrange(len(node.incoming))
        else:
            i = choice(rule_lists, old, density_factors)

        node.edge = i

        # update parent pointers
        set_parents_under(node)

        # update cut type index

        if FLAGS.type:
            node.edge = old

            for n, r in self.composed_rules_under(parent):
                CUT_INDEX.remove(r, self, n)

            node.edge = i
            for n, r in self.composed_rules_under(parent):
                CUT_INDEX.add(r, self, n)
                n.composed_rule = r

        if FLAGS.double_count:
            SAMPLER.double_count(parent)
            for cut_node in cut_nodes_under(parent):
                SAMPLER.double_count(cut_node)

        if FLAGS.check_index:
            CUT_INDEX.check(self)

    def sample_cut(self, node, parent):
        if parent is None:  # root node
            return
        if node.fj - node.fi > FLAGS.maxcut or node.ej - node.ei > FLAGS.maxcut:
            return
        rule_lists = []
        old = node.cut

        if FLAGS.double_count:
            if node.cut == 0:
                SAMPLER.double_discount(parent)
            elif node.cut == 1:
                SAMPLER.double_discount(parent)
                SAMPLER.double_discount(node)
            else:
                assert False, 'impossible'

        node.cut = 0
        rule_lists.append([self.make_composed_rule(parent)])
        node.cut = 1
        rule_lists.append([self.make_composed_rule(parent), self.make_composed_rule(node)])

        i = choice(rule_lists, old)
        node.cut = i

        if FLAGS.double_count:
            if node.cut == 0:
                SAMPLER.double_count(parent)
            elif node.cut == 1:
                SAMPLER.double_count(parent)
                SAMPLER.double_count(node)
            else:
                assert False, 'impossible'

    def type_sample_cut(self, node, parent):
        #print('type_sample_cut at node %s' % self.make_composed_rule(node))
        #print(self)
        global SITE_COUNT
        global SITE_CHECK
        global SITE_CHECK_SUCCESS
        if parent is None:  # root node
            return
        if node.sampled:
            return

        # get r1, r2, r3: r1 = r2 + r3
        old = node.cut
        if old == 0:
            r1 = parent.composed_rule
            node.cut = 1
            r2 = self.make_composed_rule(parent)
            r3 = self.make_composed_rule(node)
        else:
            r2 = parent.composed_rule
            r3 = node.composed_rule
            node.cut = 0
            r1 = self.make_composed_rule(parent)
        node.cut = old

        node_idx = index(node)

        r1_sites = []
        # intersection of r2 and r3 sites
        r2r3_sites = []

        conflict_tester = ConflictTester()
        assert conflict_tester.ok(node)

        # don't forget the current site
        if old:
            r2r3_sites.append((self, node))
        else:
            r1_sites.append((self, node))

        for s, n in CUT_INDEX.get(r3):
            SITE_CHECK += 1
            # skip root nodes
            if not hasattr(n, 'parent'):
                continue
            if n.sampled:
                continue
            p = cut_parent(n)
            if conflict_tester.ok(n) and p.composed_rule == r2:
                n.cut = 0
                if s.make_composed_rule(p) == r1:
                    r2r3_sites.append((s, n))
                    SITE_CHECK_SUCCESS += 1
                n.cut = 1

        for s, p in CUT_INDEX.get(r1):
            SITE_CHECK += 1
            if s is not self:
                n = p
                error = False
                for idx in node_idx:
                    try:
                        n = n.incoming[n.edge].tail[idx]
                    except:
                        error = True
                        break
                if error:
                    continue
                if not n.cut == 0:
                    continue
                if n.sampled:
                    continue
                n.cut = 1
                r2_t = s.make_composed_rule(p)
                r3_t = s.make_composed_rule(n)
                n.cut = 0
                if not (r2_t == r2 and r3_t == r3):
                    continue
                if conflict_tester.ok(n):
                    SITE_CHECK_SUCCESS += 1
                    r1_sites.append((s, n))

        #conflict = set(cut_zone(node, parent))
        #for sample, n, p in TYPE[self.cut_type(node, parent)]:
        #    if n not in conflict:
        #        sites.append((sample, n, p))
        #    conflict |= cut_zone(n, p)

        # check sites
        # kkk = len(r1_sites)
        # r1_sites = [(s, n) for s,n in r1_sites if check_site(s, n, r1, r2, r3)]
        # assert len(r1_sites) == kkk
        #kkk = len(r2r3_sites)
        #r2r3_sites = [(s, n) for s,n in r2r3_sites if check_site(s, n, r1, r2, r3)]
        #assert len(r2r3_sites) == kkk

        # discount
        n_r1 = len(r1_sites)
        n_r2r3 = len(r2r3_sites)
        sites = r1_sites + r2r3_sites
        n_sites = len(sites)

        # keep some stats
        SITE_COUNT[n_sites] = SITE_COUNT.get(n_sites, 0) + 1

        #print('r1: %s' % r1)
        #print('r2: %s' % r2)
        #print('r3: %s' % r3)
        #print('%s samples' % SAMPLER.nsamples())
        #print('found %s sites, %s merged, %s split' % (n_sites, n_r1, n_r2r3))

        for _ in xrange(n_r2r3):
            SAMPLER.discount(r2)
            SAMPLER.discount(r3)
        for _ in xrange(n_r1):
            SAMPLER.discount(r1)

        # number of merging sites
        if FLAGS.random_choice:
            m = random.randrange(n_sites + 1)
        else:
            m = choose_m(n_sites, (r1,), (r2, r3), SAMPLER)

        #if n_sites > 1:
        #    print('%s sites, %s merged' % (n_sites, n_r1))
        #    print('r1: %s, r2: %s, r3: %s' % (r1, r2, r3))
        #    print('merge %s sites' % m)
        # count
        for _ in xrange(m):
            SAMPLER.count(r1)
        for _ in xrange(n_sites - m):
            SAMPLER.count(r2)
            SAMPLER.count(r3)

        #print('%s samples' % SAMPLER.nsamples())

        new_r1_sites = random.sample(xrange(n_sites), m)
        r1_test = [0] * n_sites
        for i in new_r1_sites:
            r1_test[i] = 1

        for i, site in enumerate(sites):
            s, n = site
            #print('site %s' % i)
            #print(s)
            #print('at node %s %s' % (s.make_composed_rule(n), n))
            #verify_site(s, n, r1, r2, r3)
            n.sampled = True  #TODO: correct?
            # flip uncut to cut
            if i < n_r1 and not r1_test[i]:
                assert n.cut == 0
                p = cut_parent(n)
                CUT_INDEX.remove(r1, s, p)
                n.cut = 1
                CUT_INDEX.add(r2, s, p)
                CUT_INDEX.add(r3, s, n)
                p.composed_rule = r2
                n.composed_rule = r3
            # flip cut to uncut
            if i >= n_r1 and r1_test[i]:
                assert n.cut == 1
                p = cut_parent(n)
                #print('remove rule under parent')
                #print('parent %s' % p)
                #print('self %s' % n)
                CUT_INDEX.remove(r2, s, p)
                #print('remove rule under self')
                CUT_INDEX.remove(r3, s, n)
                n.cut = 0
                CUT_INDEX.add(r1, s, p)
                p.composed_rule = r1

        if FLAGS.check_index:
            CUT_INDEX.check(self)

    def sample_nt(self, node, parent):
        if node.cut and node.pnt is not None:  # sample only at cut points
            rule_lists = []

            nts = child_symbols(node.pnt)
            old = nts.index(node.nt)
            for nt in nts:
                rule_list = []
                node.nt = nt
                if parent is not None:  # not the root node
                    rule_list.append(self.make_composed_rule(parent))

                rule_list.append(self.make_composed_rule(node))
                rule_lists.append(rule_list)

            i = choice(rule_lists, old)
            node.nt = nts[i]

    def sites(self):
        queue = [(self.hg.root, None)]
        while len(queue) > 0:
            node, parent = queue.pop(0)
            yield node, parent
            if node.cut:
                parent = node
            if FLAGS.sample_cut_only:
                for child in children(node):
                    queue.append((child, parent))
            else:
                for child in node.incoming[node.edge].tail:
                    queue.append((child, parent))

    def sample(self):
        for node, parent in self.sites():
            #print('at node: %s' % self.make_composed_rule(node))
            if FLAGS.sample_level is None or node.level <= FLAGS.sample_level:
                if FLAGS.sample_edge:
                    self.sample_edge(node, parent)
                if FLAGS.sample_cut:
                    if FLAGS.type:
                        self.type_sample_cut(node, parent)
                    else:
                        self.sample_cut(node, parent)
                self.sample_nt(node, parent)

    def __str__(self):
        return self.str_helper_expand(self.hg.root)

    def str_helper(self, node, indent=0):
        result = ''
        rule = self.make_composed_rule(node)
        result += ' '*indent + str(rule) + ' ' +  str(node) + '\n'
        for child in children(node):
            result += self.str_helper(child, indent + 4)
        return result

    def str_helper_expand(self, node, indent=0):
        result = ''
        rule = self.make_one_level_rule(node)
        result += ' '*indent + str(rule) + ' ' +  str(node) + ' ' + ('cut: %s' % node.cut) + '\n'
        for child in node.incoming[node.edge].tail:
            result += self.str_helper_expand(child, indent + 4)
        return result

if __name__ == '__main__':
    gflags.DEFINE_integer(
        'interval',
        5000,
        'Print stat every #interval# sentences.')
    gflags.DEFINE_integer(
        'iter',
        1,
        'Number of sampling iterations.')
    gflags.DEFINE_integer(
        'dump_iter',
        1,
        'Dump trees every #dump_iter# iterations.')
    gflags.DEFINE_string(
        'dump',
        'dump',
        'Dump directory.')

    try:
        argv = FLAGS(sys.argv)  # parse flags
    except gflags.FlagsError as e:
        print('%s\nUsage: %s ARGS\n%s' % (e, sys.argv[0], FLAGS))
        sys.exit(1)

    if FLAGS.seed_random:
        random.seed(0)

    if FLAGS.base == 'poisson':
        base = poisson
    elif FLAGS.base == 'abtest':
        base = abtest_base

    ffilename = argv[1]
    efilename = argv[2]
    afilename = argv[3]
    ffile = open(ffilename)
    efile = open(efilename)
    afile = open(afilename)
    alignments = alignment.Alignment.reader_pharaoh(ffile, efile, afile)

    LEXICAL_WEIGHTER = LexicalWeighter()

    os.system('rm -rf %s' % FLAGS.dump)
    os.mkdir(FLAGS.dump)
    logger.file = open(os.path.join(FLAGS.dump, 'log'), 'w')
    flagfile = open(os.path.join(FLAGS.dump, 'run.flag'), 'w')
    flagfile.write(FLAGS.FlagsIntoString())
    flagfile.close()

    SAMPLER = None

    # initialization
    samples = []
    logger.writeln()
    logger.writeln('read samples')
    for i, a in enumerate(timed(alignments), 1):
        hg, a = phrase_decomposition_forest(a)
        samples.append(Sample(hg, a))
    ffile.close()
    efile.close()
    afile.close()

    if not FLAGS.refine:
        FLAGS.split_at_iter_0 = False

    CUT_INDEX = CutTypeIndex(samples)

    init_split(samples, FLAGS.split_at_iter_0)

    logger.writeln('%s rules, %s rule types, loglikelihood: %s' %
                   (SAMPLER.nsamples(), SAMPLER.ntypes(), SAMPLER.likelihood()))

    dump_trees(samples, os.path.join(FLAGS.dump, 'iter-0000'))

    # sampling
    iteration = 1
    iter_start = time.time()
    if FLAGS.level_inc is not None:
        FLAGS.sample_level = 1
    while iteration <= FLAGS.iter:
        #print('iteration %s' % iteration)
        logger.writeln()
        logger.writeln('iteration %s' % iteration)
        if FLAGS.sample_level is not None:
            logger.writeln('level <= %s' % FLAGS.sample_level)

        if FLAGS.level_inc is not None and iteration % FLAGS.level_inc == 0:
                FLAGS.sample_level += 1

        if FLAGS.refine and iteration % FLAGS.split_iter == 0:
            init_split(samples)

        # extra init for type-based sampling
        if FLAGS.type:
            for sample in samples:
                for node in sample.hg:
                    node.sampled = False
            # stat of site numbers
            SITE_COUNT = {}
            SITE_CHECK = 0
            SITE_CHECK_SUCCESS = 0

        if FLAGS.model == 'PY' and FLAGS.discount > 0:
            SAMPLER.update_rule_size_tables()

        for s in timed(samples):
            #print(s)
            s.sample()

        logger.writeln('iteration time: %s sec' % (time.time() - iter_start))
        logger.writeln('%s rules, %s rule types, loglikelihood: %s' %
                       (SAMPLER.nsamples(), SAMPLER.ntypes(), SAMPLER.likelihood()))
        logger.writeln('memory: %s' % memory())
        logger.writeln('resident memory: %s' % resident())

        if FLAGS.type:
            logger.writeln('%s sampling operations in total, distribution of number of sites: %s' % (sum(SITE_COUNT.values()), SITE_COUNT))
            logger.writeln('%s sites: %s singleton sites, %s (2-10) sites, %s (>10) sites' % (sum(k*v for k,v in SITE_COUNT.items()),
                                                                                              sum(k*v for k,v in SITE_COUNT.items() if k==1),
                                                                                              sum(k*v for k,v in SITE_COUNT.items() if 2<=k<=10),
                                                                                              sum(k*v for k,v in SITE_COUNT.items() if k>10)))

            logger.writeln('site checks: %s, success: %s' % (SITE_CHECK, SITE_CHECK_SUCCESS))
        if iteration % FLAGS.dump_iter == 0:
            dump_trees(samples,
                       os.path.join(FLAGS.dump, 'iter-%s' % str(iteration).rjust(4, '0')))

        iter_start = time.time()
        iteration += 1
