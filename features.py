from common import INF
from decode import Deduction, Item
import feature as feature_lib
import logger
import gflags
from rule import is_virtual
FLAGS = gflags.FLAGS

class Features(object):
    """A list of features used by the decoder to compute scores of newly
    generated edges"""
    def __init__(self, features_weights, weight_range_str=None):
        # list of stateful features, which take a deduction to compute
        self.stateful = []
        # list of stateless features, which take only a rule to compute
        self.stateless = []
        self.stateful_feature_index = 0

        self.decoder = None  # the current decoder using this set of features

        feature_list, weights = self.load_features(features_weights)
        for f, w in zip(feature_list, weights):
            self.add(f, w)

        if weight_range_str is None:
            self.weight_ranges = [(-INF, INF)] * len(self.get_features())
        else:
            self.weight_ranges = self.parse_weight_ranges(weight_range_str)
            assert(len(self.weight_ranges) == len(self.get_features()))

    def get_features(self):
        return self.stateful + self.stateless

    def update_weights(self):
        pass

    def make_new_item(self, rule, ants, i, j):
        "given rule and items, derive new item"
        deduction = Deduction(rule, ants, self)
        dcost = rule.cost  # precomputed cost of stateless features
        newstates = []
        for f, weight in self.stateful:
            fcost, state = f.weight(deduction)
            dcost += fcost * weight
            deduction.fcosts.append(fcost)
            newstates.append(state)
        # append stateless features
        deduction.fcosts += rule.fcosts
        cost = dcost + sum(ant.cost for ant in ants) # total new item cost
        deduction.cost = dcost
        newitem =  Item(rule.lhs, i, j, tuple(newstates), deduction, cost)
        # heuristic for an item
        newitem.hcost = rule.hcost
        # LM heuristic. disabled for the moment.
        # if newitem.j == self.decoder.N:
        #     newitem.rightmost = True
        # else:
        #     newitem.rightmost = False
        # for f, weight in self.stateful + self.stateless:
        #     h = f.heuristic(newitem)
        #     newitem.hcost += h
        return newitem

    def score_rule(self, rule):
        """computing feature scores for a rule, giving the rule 'cost' and
        'fcosts' fields. this uses only stateless features """
        cost = 0
        fcosts = []
        for feature, weight in self.stateless:
            fcost = feature.weight(rule)
            cost += fcost*weight
            fcosts.append(fcost)
        # the cost of virtual rules are used as a heuristic only
        if is_virtual(rule.lhs):
            rule.cost = 0
            rule.hcost = cost
            rule.fcosts = [0] * len(fcosts)
        else:
            rule.cost = cost
            rule.fcosts = fcosts
            rule.hcost = 0

    # ----------- begin of methods class users usually do not need------------

    def load_features(self, features_weights):
        features = []
        weights = []
        for s in features_weights:
            feature_name, weight_str = s.split(':')
            weight = float(weight_str)
            feature_class = getattr(feature_lib, feature_name, None)
            if feature_class is None:
                logger.writeln('unknown feature: %s' % feature_name)
            else:
                if feature_name.endswith('LM'):
                    feature = feature_class(FLAGS.lm_order, FLAGS.lm)
                else:
                    feature = feature_class()
                features.append(feature)
                weights.append(weight)
        return features, weights

    def parse_weight_ranges(self, weight_range_str):
        features_lbounds_ubounds = weight_range_str.split()
        features = features_lbounds_ubounds[::3]
        lbounds = features_lbounds_ubounds[1::3]
        ubounds = features_lbounds_ubounds[2::3]
        name2bounds = dict((f, (eval(lb), eval(ub)))
                           for f,lb,ub in zip(features, lbounds, ubounds))
        result = []
        for f,w in self.get_features():
            result.append(name2bounds[f.name])
        return result

    def add(self, feature, weight):
        """give an index to stateful features so that the feature know which
        state in an item's state list is the feature's state"""
        if feature.stateless:
            self.stateless.append((feature, weight))
        else:
            feature.i = len(self.stateful)
            self.stateful.append((feature, weight))
