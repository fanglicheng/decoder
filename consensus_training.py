#!/usr/bin/env python3

import os

from ngram import NgramEnumerator
from common import INF
import hypergraph
import logprob
import logger
from hypergraph import Node, Deserializer
from decode import Deduction

def cartesian(l):
    """given a list of k containers, return all k tuples made of one element
    from each container.
            
    >>> [r for r in cartesian([[1,2,3],[4,5,6],[7,8,9]])]
    [(1, 4, 7), (1, 4, 8), (1, 4, 9), (1, 5, 7), (1, 5, 8), (1, 5, 9), (1, 6, 7), (1, 6, 8), (1, 6, 9), (2, 4, 7), (2, 4, 8), (2, 4, 9), (2, 5, 7), (2, 5, 8), (2, 5, 9), (2, 6, 7), (2, 6, 8), (2, 6, 9), (3, 4, 7), (3, 4, 8), (3, 4, 9), (3, 5, 7), (3, 5, 8), (3, 5, 9), (3, 6, 7), (3, 6, 8), (3, 6, 9)]
"""
    if not l:
        yield ()
        return
    for elem in l[0]:
        for result in cartesian(l[1:]):
            yield (elem,) + result

class NgramCounter:
    def __init__(self, max_n):
        self.max_n = max_n
        self.enums = [NgramEnumerator(i+1) for i in range(max_n)]

    def count(self, hg):
        pass

    def mark_ngrams(self, hg):
        """take a Item/Deduction hypergraph, add ngram features to each edge
        ref_count is a RefCounter
        """
        for node in hg.topo_order():
            # ngrams[i] on a node saves all i+1 gram decorations
            node.ngrams = [set() for i in range(self.max_n)]
            for edge in node.incoming:
                # ngrams[i] on a edge counts all i+1 grams
                edge.ngrams = [{} for i in range(self.max_n)]

        for i in range(self.max_n):
            enum = self.enums[i]
            for node in hg.topo_order():
                #print(node)
                for edge in node.incoming:
                    #print(edge)
                    for states in cartesian([tail_node.ngrams[i]
                                             for tail_node in edge.tail]):
                        s = tuple(edge.rule.rewrite(states))
                        node.ngrams[i].add(enum.elide(s))
                        for ngram in enum.ngrams(s):
                            if ngram in edge.ngrams[i]:
                                edge.ngrams[i][ngram] += 1
                            else:
                                edge.ngrams[i][ngram] = 1

class ConsensusTrainingCounter(object):
    """one counter each sentence"""
    def __init__(self,
                 trainer,
                 hg_file,
                 ref,
                 max_n,
                 feature_n):
        self.trainer = trainer
        self.hg_file = hg_file
        self.ref = ref
        self.max_n = max_n
        self.feature_n = feature_n
        self.count_ngrams()

    def count_ngrams(self):
        deserializer = Deserializer(Node, Deduction)
        self.hg = deserializer.deserialize(self.hg_file)
        for edge in self.hg.edges():
            edge.ngram_counts = []
            edge.ngram_clipped_counts = []
            for i in range(self.max_n):
                count = sum(c for c in edge.ngrams[i].values())
                clipped_count = sum(min(c, self.ref[ngram])
                                    for ngram, c in edge.ngrams[i].items())
                edge.ngram_counts.append(count)
                edge.ngram_clipped_counts.append(clipped_count)

    def update(self):
        """update edge cost and do the computations necessary for both cobleu
        and gradient"""
        self.update_edge_costs()
        self.hg.set_semiring(hypergraph.LOGPROB)
        self.hg.set_functions(lambda d: -d.cost, None, None)
        self.compute_edge_posterior()
        self.compute_expected_ngram_counts()
        self.ref_length = self.ref.closest_length(self.expected_counts[0])

    def update_edge_costs(self):
        for edge in self.hg.edges():
            edge.cost = sum(f*w 
                            for f,w in zip(edge.fcosts, self.trainer.weights))

    def compute_expected_ngram_counts(self):
        self.expected_clipped_counts = []
        self.expected_counts = []
        for n in range(self.max_n):
            count = self.expected_count(self.ngram_count(n))
            clipped_count = self.expected_count(self.ngram_clipped_count(n))
            # print('%s-gram: %s/%s' % (n+1, clipped_count, count))
            self.expected_clipped_counts.append(clipped_count)
            self.expected_counts.append(count)

    def compute_expected_feature_counts(self):
        self.expected_feature_counts = []
        for i in range(self.feature_n):
            feature_count = self.expected_count(self.decoder_feature(i))
            # print('feature %s: %s' % (i, feature_count))
            self.expected_feature_counts.append(feature_count)

    def compute_expected_products(self):
        # print('clipped ngram count')
        self.ep_clipped_count_feature = []
        for n in range(self.max_n):
            # print('%s gram' % (n+1))
            self.compute_edge_expectation(self.log(
                                          self.ngram_clipped_count(n)))
            products = []
            for i in range(self.feature_n):
                # print('feature %s' % i)
                product = self.expected_product(self.decoder_feature(i))
                products.append(product)
                # print(product)
            self.ep_clipped_count_feature.append(products)
        # print('ngram count')
        self.ep_count_feature = []
        for n in range(self.max_n):
            # print('%s gram' % (n+1))
            self.compute_edge_expectation(self.log(self.ngram_count(n)))
            products = []
            for i in range(self.feature_n):
                # print('feature %s' % i)
                product = self.expected_product(self.decoder_feature(i))
                products.append(product)
                # print(product)
            self.ep_count_feature.append(products)

    # counters 
    def compute_edge_posterior(self): 
        self.edge_posterior = []
        self.hg.inside()
        self.hg.outside()
        self.normalization = self.hg.root.inside
        #TODO: normalization can be done later
        for edge in self.hg.edges():
            posterior = logprob.eexp(edge.posterior() - self.normalization)
            self.edge_posterior.append(posterior)

    def expected_count(self, f):
        """f is a function of edge"""
        result = 0
        for eid, edge in enumerate(self.hg.edges()):
            result += self.edge_posterior[eid]*f(edge)
        return result

    def compute_edge_expectation(self, f):
        self.edge_expectation = []
        self.hg.f = f
        self.hg.inside_exp()
        self.hg.outside_exp()
        for edge in self.hg.edges():
            exp = logprob.eexp(edge.expectation() - self.normalization)
            self.edge_expectation.append(exp)

    def expected_product(self, f):
        """f is a function of edge"""
        result = 0
        for eid, edge in enumerate(self.hg.edges()):
            result += self.edge_expectation[eid]*f(edge)
        return result

    # feature function generators
    def ngram_count(self, n):
        return lambda edge: edge.ngram_counts[n]

    def ngram_clipped_count(self, n):
        return lambda edge: edge.ngram_clipped_counts[n]

    def decoder_feature(self, i):
        return lambda edge: -edge.fcosts[i]

    def log(self, f):
        """return a function which is log f"""
        return lambda x: logprob.elog(f(x))
     
class ConsensusTrainer(object):
    def __init__(self, max_n, features, hgdir, refs):
        self.max_n = max_n
        self.features = features.get_features()
        self.feature_n = len(self.features)
        self.weights = [w for f,w in self.features]
        self.weight_ranges = features.weight_ranges
        self.old_cobleu = -INF
        self.cobleu = -INF
        self.converge_thres = 0.001 # TODO: magic number
        self.step_size = sum(w for f,w in self.features)/self.feature_n

        hg_files = [f for f in os.listdir(hgdir) if f.startswith('hg_')]
        
        self.counters = []
        for filename in hg_files:
            prefix, jid = filename.split('_')
            jid = int(jid)
            training_counter = ConsensusTrainingCounter(
                self,
                '%s/%s' % (hgdir, filename),
                refs.get_counter(jid-1),
                self.max_n,
                self.feature_n)
            self.counters.append(training_counter)

    def optimize(self):
        i = 0
        old_cobleu = -INF
        self.update()
        while True:
            if logger.level >= 1:
                logger.writeln()
                logger.writeln('step %s' % i)    
            gradient = self.compute_gradient()
            old_weights = [w for w in self.weights]

            # line search
            while True:
                self.weights = self.new_weights(old_weights,
                                                gradient,
                                                self.step_size)
                # self.normalize_weights()
                if logger.level >= 1:
                    logger.writeln('new weights: %s' % self.weights)
                if self.out_of_range(self.weights):
                    if logger.level >= 1:
                        logger.writeln('weights out of range')
                    self.step_size /= 2
                    continue
                self.update()
                cobleu = self.compute_cobleu()
                if cobleu > old_cobleu:
                    if logger.level >= 1:
                        logger.writeln('enlarge step size')
                    self.step_size *= 2
                    break
                else:
                    # shrink step if overshoot
                    if logger.level >= 1:
                        logger.writeln('shrink step size')
                    self.step_size /= 2

            if cobleu - old_cobleu < self.converge_thres:
                break
            else:
                old_cobleu = cobleu
            # dangerous, this resets the semiring
            # self.write_sentence()
            i += 1

    # ----------- begin of methods class users usually do not need------------

    def compute_gradient(self):
        self.collect_expected_feature_counts()
        self.collect_expected_products()
        result = []
        for i in range(self.feature_n):
            # print('feature %s' % i)
            gradient = 0

            tmp = 0
            for n in range(self.max_n):
                # print('clip count %s-gram gradient' % (n+1))
                if self.expected_clipped_counts[n] == 0:
                    continue
                clipped_count_grad = \
                        self.ep_clipped_count_feature[n][i] - \
                        self.expected_clipped_counts[n] * \
                        self.expected_feature_counts[i]
                # print(clipped_count_grad)
                tmp += clipped_count_grad / self.expected_clipped_counts[n]
            gradient += tmp/self.max_n

            tmp = 0
            for n in range(self.max_n):
                # print('count %s-gram gradient' % (n+1))
                if self.expected_counts[n] == 0:
                    continue
                count_grad = self.ep_count_feature[n][i] - \
                        self.expected_counts[n]*self.expected_feature_counts[i]
                # print(count_grad)
                tmp += count_grad / self.expected_counts[n]
            gradient -= tmp/self.max_n

            # brevity penalty
            if self.expected_counts[0] < self.ref_length:
                gradient += (self.ep_count_feature[0][i] -
                             self.expected_counts[0]* \
                             self.expected_feature_counts[i]) * \
                             self.ref_length / \
                        (self.expected_counts[0])^2

            result.append(gradient)
        if logger.level >= 1:
            logger.writeln('gradient: %s' % result)
        return result

    def compute_cobleu(self):
        if logger.level >= 1:
            logger.writeln()
            logger.writeln('----CoBLEU----')
        cobleu = 0
        for n in range(self.max_n):
            if logger.level >= 1:
                logger.writeln('%s-gram: %s/%s' % \
                               (n+1,
                                self.expected_clipped_counts[n],
                                self.expected_counts[n]))
            precision = self.expected_clipped_counts[n] / self.expected_counts[n]
            cobleu += logprob.elog(precision)
        cobleu /= self.max_n
        # brevity
        cobleu += min(0, 1 - self.ref_length / self.expected_counts[0])
        if logger.level >= 1:
            logger.writeln('---- %s ----' % cobleu)
        return cobleu

    def update(self):
        """update edge cost and do the computations necessary for both cobleu
        and gradient"""
        for counter in self.counters:
            counter.update()
        self.collect_expected_ngram_counts()
        self.collect_ref_length()

    def collect_expected_ngram_counts(self):
        self.expected_clipped_counts = [0] * self.max_n
        self.expected_counts = [0] * self.max_n
        for counter in self.counters:
            self.expected_clipped_counts = \
                    self.listsum(self.expected_clipped_counts,
                                 counter.expected_clipped_counts)
            self.expected_counts = \
                    self.listsum(self.expected_counts,
                                 counter.expected_counts)

    def collect_ref_length(self):
        self.ref_length = 0
        for counter in self.counters:
            self.ref_length += counter.ref_length

    def collect_expected_feature_counts(self):
        self.expected_feature_counts = [0] * self.feature_n
        for counter in self.counters:
            counter.compute_expected_feature_counts()
            self.expected_feature_counts = \
                    self.listsum(self.expected_feature_counts,
                                 counter.expected_feature_counts)

    def collect_expected_products(self):
        self.ep_clipped_count_feature = [[0]*self.feature_n
                                         for n in range(self.max_n)]
        self.ep_count_feature = [[0]*self.feature_n
                                 for n in range(self.max_n)]
        for counter in self.counters:
            counter.compute_expected_products()
            for n in range(self.max_n):
                self.ep_clipped_count_feature[n] = \
                        self.listsum(self.ep_clipped_count_feature[n],
                                     counter.ep_clipped_count_feature[n])
                self.ep_count_feature[n] = \
                        self.listsum(self.ep_count_feature[n],
                                     counter.ep_count_feature[n])

    # def write_sentence(self):
    #     self.hg.set_semiring(hypergraph.SHORTEST_PATH)
    #     self.hg.set_functions(lambda x: x.cost, None, None)
    #     out = ' '.join(x 
    #                    for x in self.hg.root.best_paths()[0].translation[1:-1])
    #     if logger.level >= 1:
    #         logger.writeln(out)

    def new_weights(self, weights, direction, stepsize):
        return [w + stepsize*d for w,d in zip(weights, direction)]

    def normalize_weights(self):
        s = sum(w for w in self.weights)
        self.weights = [w/s for w in self.weights]

    def listsum(self, l1, l2):
        return [x1 + x2 for x1, x2 in zip(l1, l2)]

    def out_of_range(self, weights):
        result = False
        for w, wrange in zip(weights, self.weight_ranges):
            lbound, ubound = wrange
            if not lbound <= w <= ubound:
                result = True
                break
        return result

if __name__ == '__main__':
    import doctest
    doctest.testmod()
