from heapq import heapify, heappush, heappop

import gflags
FLAGS = gflags.FLAGS
import hypergraph
from hypergraph import Edge, Node, Hypergraph
from rule import Rule, is_virtual, nocat
from chart import Chart
from cube import Cube
import logger
from grammar import LexicalITG
from common import cyk_spans
from dot_chart import DotChart
import decoding_flags

gflags.DEFINE_integer(
    'bin_size',
    10,
    'Bin size in top-K pruning.')
gflags.DEFINE_string(
    'unknown_nonterminal',
    '[A]',
    'Nonterminal name used for rules handling out-of-vocabulary words.')
gflags.DEFINE_boolean(
    'pass_unknown_words',
    False,
    'Pass unknown words into output instead of deleting.')
gflags.DEFINE_string(
    'start_symbol',
    '[S]',
    'Start symbol.')
gflags.DEFINE_string(
    'goal_symbol',
    '[GOAL]',
    'Goal symbol.')

class Deduction(Edge):
    def __init__(self, rule=None, ants=None, features=None):
        # rule: the rule from which this deduction is genrated
        # ants: antecedent items
        # features: the Feature object used to score this deduction
        Edge.__init__(self)
        if ants is not None:
            for ant in ants:
                self.add_tail(ant)
        self.rule = rule
        self.features = features
        # list of feature values, first stateful features,
        # then stateless features
        self.fcosts = []
        self.cost = 0

    def __str__(self):
        if self.features is None:
            feature_str = ','.join('%.2f' % fcost for fcost in self.fcosts)
        else:
            flist = self.features.stateful + self.features.stateless
            feature_str = ','.join('%s:%.2f' % (f.name, fcost)
                                   for (f,w), fcost in zip(flist, self.fcosts))
        return 'Deduction: %.2f (%s) [rule: %s]' % (self.cost,
                                                    feature_str,
                                                    self.rule)

    def make_path(self, subpaths):
        """Extends the base class make_path method to generate a Path object
        attached with:
            1) weight, (done in base class)
            2) translation hypothesis,
            3) list of accumulated feature values"""
        path = Edge.make_path(self, subpaths)
        if FLAGS.preprocess:
            self.rule.align_special_symbols()
        path.composed_rule = self.rule.compose([p.composed_rule for p in subpaths])
        path.translation = path.composed_rule.e
        path.fcosts = list(self.fcosts)  # copy 
        for p in subpaths:
            if p is not None:
                for i, fcost in enumerate(p.fcosts):
                    path.fcosts[i] += fcost
        return path

    def serialize(self):
        """extends hypergraph.Edge.serialize()"""
        edge_str = Edge.serialize(self)
        if hasattr(self, 'ngrams'):
            ngram_str = ''
            for i in range(len(self.ngrams)):
                for ngram, count in self.ngrams[i].items():
                    ngram_str += '%s $%s$ ' % (' '.join(word for word in ngram),
                                               count)
            feature_str = ' '.join(str(fcost) for fcost in self.fcosts)
            deduction_str = ' ||| '.join([edge_str, ngram_str, feature_str])
            result = ' ||||| '.join([deduction_str, str(self.rule)])
        else:
            result = '%s ||||| %s' % (edge_str, self.rule)
        return result

    def deserialize(self, line):
        deduction_str, rule_str = line.split('|||||')
        #edge_str, ngram_str, feature_str = deduction_str.split('|||')
        edge_str = deduction_str

        # init with Edge deserialization
        tail_ids, head_id = Edge.deserialize(self, edge_str)
        # load ngrams
        max_n = 3  # TODO: remove magic number
        self.ngrams = [{} for i in range(max_n)]
        wordlist = []
        #for token in ngram_str.split():
        #    if len(token) >= 3 and \
        #       token.startswith('$') and \
        #       token.endswith('$'):  # end of ngram
        #        count = int(token[1:-1])
        #        ngram = tuple(wordlist)
        #        wordlist = []
        #        self.ngrams[len(ngram)-1][ngram] = count
        #    else:  # add another word to current ngram 
        #        wordlist.append(token)
        #
        # load feature costs 
        #self.fcosts = []
        #for fcost in feature_str.split():
        #    self.fcosts.append(float(fcost)) 
        # load rule
        self.rule = Rule()
        self.rule.fromstr(rule_str)

        return tail_ids, head_id
        
class Item(Node):
    def __init__(self, var, i, j, state, deduction, cost):
        Node.__init__(self)
        self.var = var
        self.i = i
        self.j = j
        self.state = state
        self.cost = cost
        self.hcost = 0  # heuristic
        self.add_incoming(deduction)
        self.closed = False
        self.dead = False  # pruned by Bin

    def rank_cost(self): 
        return self.cost + self.hcost

    def __eq__(self, other):
        # within a Bin that contains killed items (None)
        # sometimes an item is compared with None
        # TODO: changed this with a heap that supports
        # decrease-key operation
        if other is None:
            return True
        return self.var == other.var and \
               self.i == other.i and \
               self.j == other.j and \
               self.state == other.state

    def __hash__(self):
        return hash( (self.var, self.i, self.j, self.state) )

    def __lt__(self, other):
        return self.rank_cost() < other.rank_cost()

    def __str__(self):
        return '[%s,%s,%s,%s,%.2f (%.2f+%.2f)]' % \
                (self.var,
                 self.i,
                 self.j,
                 self.state,
                 self.rank_cost(),
                 self.cost,
                 self.hcost)

    def goal(self):
        return self.var == FLAGS.goal_symbol

    def merge(self, item):
        "return true if self is merged into other item, false otherwise"
        assert self == item
        if self.rank_cost() > item.rank_cost():  # item better
            for edge in self.incoming:
                item.add_incoming(edge)
            self.dead = True
            return True
        else:  # self better
            for edge in item.incoming:
                self.add_incoming(edge)
            item.dead = True
            return False

    def unary_cycle(self):
        """called when this item is first generated to determine if there
        is a unary cycle."""
        result = False
        if self.incoming[0].rule.arity == 1:
            queue = [self.incoming[0].tail[0]]
            found = set(queue)
            while queue:
                item = queue.pop(0)
                if item == self:
                    result = True
                    break
                for deduction in item.incoming:
                    if deduction.rule.arity == 1:
                        child = deduction.tail[0]
                        if child not in found:
                            queue.append(child)
        return result

def cyk_heuristic(item):
    return item.j - item.i

def best_first_heuristic(item):
    return item.rank_cost()

class Agenda(object):
    def __init__(self, heuristic='cyk'):
        self.items = []
        if heuristic == 'cyk':
            self.heuristic = cyk_heuristic
        elif heuristic == 'best_first':
            self.heuristic = best_first_heuristic
        else:
            assert False, 'unknown agenda decoding heuristic'
        
        # statistics
        self.pushed = 0
        self.popped = 0
        self.deadpop = 0

    def push(self, item):
        if logger.level >= 4:
            logger.writeln('push:')
            logger.writeln(item)
            logger.writeln(item.incoming[0])
        h = self.heuristic(item)
        heappush(self.items, (h, item) )
        self.pushed += 1

    def pop(self):
        "Return None if agenda is empty"
        while True:
            try:
                h, item = heappop(self.items)
            except IndexError:  # empty heap, return None
                break
            if item.dead:  # item pruned in chart
                if logger.level >= 5:
                    logger.writeln('pop dead item: %s' % item)
                    logger.writeln(item)
                    logger.writeln(item.incoming[0].rule)
                self.deadpop += 1
            else:
                if logger.level >= 4:
                    logger.writeln('pop: %s' % item)
                    logger.writeln(item.incoming[0].rule)
                self.popped += 1
                return item

    def __len__(self):
        return len(self.items)

class Decoder(object):
    def __init__(self,
                 fwords,
                 grammars,
                 features):
        self.N = len(fwords)
        self.chart = Chart(fwords, FLAGS.start_symbol)
        self.agenda = Agenda()
        self.fwords = fwords
        self.grammars = grammars
        self.features = features
        self.features.decoder = self

        self.dotcharts = []

        # statistics
        self.closed = 0
        self.neighbors_tried = 0
        self.neighbors_closed = 0
        self.cubes_built = 0
        self.unary_edges_proposed = 0
        self.nonunary_edges_proposed = 0

    def decode_earley(self):
        """Returns None if no goal item found"""
        self.initialize_earley()
        for i, j in cyk_spans(self.N): 
            if logger.level >= 4:
                logger.writeln()
                logger.writeln('---- span (%s %s) ----' % (i, j))
            # finish dot chart, build a cube
            new_items = Cube()
            new_virtual_items = Cube()
            for dotchart in self.dotcharts:
                if logger.level >= 4:
                    logger.writeln()
                    logger.writeln('dot chart for %s' % dotchart.grammar.name)
                dotchart.expand(i, j)
                for dotitem in dotchart.bins[i][j]:
                    if dotitem.node.filled:
                        for lhs, rulebin in dotitem.node.iter_rulebins():
                            bins = (rulebin,) + dotitem.ants
                            if is_virtual(lhs):
                                new_virtual_items.add_cube(
                                    bins, self.get_cube_op(i, j))
                            else:
                                new_items.add_cube(
                                    bins, self.get_cube_op(i, j))
                            self.cubes_built += 1
            if logger.level >= 4:
                logger.writeln(' -- cubes --')
                logger.writeln(new_items)
                logger.writeln(' -- cubes for virtual items--')
                logger.writeln(new_virtual_items)
            # pop new items from the cube
            for cube in [new_items, new_virtual_items]:
                for new_item in cube.iter_top(FLAGS.bin_size):
                    self.nonunary_edges_proposed += 1
                    if logger.level >= 4:
                        logger.writeln('cube pop: %s' % new_item)
                        logger.writeln(new_item.incoming[0])
                    added = self.chart.add(new_item)
                    if logger.level >= 4:
                        logger.writeln('added: %s' % added)
            # apply unary rules
            self.unary_expand(i, j)
            # generate dot items like A->B.C (first nonterminal matched)
            # after the unary derivations are all finished
            for dotchart in self.dotcharts:
                if logger.level >= 4:
                    logger.writeln()
                    logger.writeln('unary expand for dot chart %s' %
                                   dotchart.grammar.name)
                dotchart.unary_expand(i, j)
        return self.get_goal()

    def decode_cyk(self):
        """Returns None if no goal item found"""
        self.initialize()
        for i, j in cyk_spans(self.N):
            self.binary_expand(i, j)
            self.unary_expand(i, j)
        return self.get_goal()

    def decode(self):
        """Returns None if no goal item found"""
        self.initialize()
        while len(self.agenda) > 0:
            item = self.agenda.pop()
            if item is None:  # agenda empty
                return
            if item.goal(): # goal item
                return item
            if item.closed:
                continue
            item.closed = True
            self.closed += 1
            # binary
            for litem in self.chart.leftneighbors(item):
                self.neighbors_tried += 1
                if litem.closed:
                    self.neighbors_closed += 1
                    for newitem in self.deduce([litem, item]):
                        if self.chart.add(newitem):
                            self.agenda.push(newitem)
            for ritem in self.chart.rightneighbors(item):
                self.neighbors_tried += 1
                if ritem.closed:
                    self.neighbors_closed += 1
                    for newitem in self.deduce([item, ritem]):
                        if self.chart.add(newitem):
                            self.agenda.push(newitem)
            for newitem in self.deduce([item]):
                if self.chart.add(newitem):
                    self.agenda.push(newitem)

    def agenda_stats(self):
        header = '{:-^50}\n'
        field = '{:<35}{:>15}\n'

        result = header.format('Decoding Stats')
        result += field.format('[non-unary edges proposed]:',
                               self.nonunary_edges_proposed)
        result += field.format('[unary edges proposed]:',
                               self.unary_edges_proposed)
        result += field.format('[total edges proposed]:',
                               self.unary_edges_proposed + 
                               self.nonunary_edges_proposed)
        result += field.format('[cubes (non-unary -LM edges)]:',
                               self.cubes_built)
        result += '\n'
        result += header.format('Agenda Stats')
        result += field.format('[pushed]:',
                               self.agenda.pushed)
        result += field.format('[popped]:',
                               self.agenda.popped)
        result += field.format('[dead pop]:',
                               self.agenda.deadpop)
        result += field.format('[closed]:',
                               self.closed)
        result += field.format('[final agenda size]:',
                               len(self.agenda))
        result += field.format('[neighbors closed]:',
                               self.neighbors_closed)
        result += field.format('[neighbors tried]:',
                               self.neighbors_tried)
        return result

    # ----------- begin of methods class users usually do not need------------

    def initialize_earley(self):
        for grammar in self.grammars:
            dotchart = DotChart(self.chart, grammar)
            self.dotcharts.append(dotchart)
        # handling unknown words
        for item in self.iter_unknown_items():
            self.chart.add(item)

    def initialize(self):
        "use lexical grammar to initialize"
        for grammar in self.grammars:
            self.initialize_with_lexgrammar(grammar.lexgrammar)
        # handling unknown words
        for item in self.iter_unknown_items():
            self.agenda.push(item)
            self.chart.add(item)

    def initialize_with_lexgrammar(self, lexgrammar):
        """initialize chart and agenda with a pure lexical grammar"""
        # items in lexchart are pointers to TrieNodes in lexgrammar
        lexchart = [[None for i in range(self.N+1)] for j in range(self.N+1)]
        # seed the chart
        for i in range(self.N):
            lexchart[i][i] = lexgrammar.root
        #TODO: insertion?
        for i, j in cyk_spans(self.N):
            prevnode = lexchart[i][j-1]      
            word = self.fwords[j-1]          
            if prevnode and word in prevnode:
                curnode = lexchart[i][j-1][word] # scan one word
                lexchart[i][j] = curnode
                for rule in curnode.iter_rules():
                    item = self.features.make_new_item(rule, (), i, j)
                    self.agenda.push(item)
                    self.chart.add(item)

    def unknown_word_rule(self, word):
        """generated a rule for a unknown word"""
        # delete or pass?
        rule = Rule()
        if FLAGS.pass_unknown_words:
            rule.fromstr('%s ||| %s ||| %s ||| 1 1 1 1 1' %
                         (FLAGS.unknown_nonterminal,
                          word,
                          word))
        else:
            rule.fromstr('%s ||| %s ||| %s ||| 1 1 1 1 1' %
                         (FLAGS.unknown_nonterminal,
                          word,
                          ''))
        rule.is_unknown = True  # for computing feature scores
        self.features.score_rule(rule)
        return rule

    def deduce(self, items):
        """given one or two item(s), try applying all possible ITG rules and
        yield the deduced items"""
        for grammar in self.grammars:
            for item in self.deduce_with_itg(grammar.itg, items):
                if logger.level >= 4:
                    logger.writeln('new item:')
                    logger.writeln(item)
                    logger.writeln(item.incoming[0])
                yield item

    def deduce_with_itg(self, itg, items):
        # unary
        if len(items) == 1:
            item = items[0]
            for new_item in self.unary_deduce(item):
                yield new_item
        # binary
        elif len(items) == 2:
            item1, item2 = items
            sym_list = [item.var for item in items]
            for rule in itg.iter_rules(sym_list):
                # derive start symbol only for spans [0,i]
                if rule.lhs == FLAGS.start_symbol and item1.i != 0:
                    continue
                new_item =  self.features.make_new_item(rule,
                                                        items,
                                                        item1.i,
                                                        item2.j)
                yield new_item

    def binary_expand(self, i, j):
        if logger.level >= 4:
            logger.writeln('span %s %s' % (i, j))
        new_items = Cube()
        for k in range(i+1, j):
            for lvar, lbin in self.chart.iter_items_by_nts(i, k):
                for rvar, rbin in self.chart.iter_items_by_nts(k, j):
                    for grammar in self.grammars:
                        rulebin = grammar.itg.get_sorted_rules((lvar, rvar))
                        if rulebin:
                            new_items.add_cube((rulebin, lbin, rbin),
                                               self.get_cube_op(i,j))
        for new_item in new_items.iter_top(FLAGS.bin_size):
            if logger.level >= 4:
                logger.writeln(new_item)
            self.chart.add(new_item)

    def unary_expand(self, i, j):
        if logger.level >= 4:
            logger.writeln()
            logger.writeln('-- Unary expand (%s,%s) --'% (i,j))
        queue = []
        for var, bin in self.chart.iter_items_by_nts(i, j):
            for item in bin:
                queue.append(item)
        heapify(queue)
        while queue:
            item = heappop(queue)
            if logger.level >= 4:
                logger.writeln('unary queue pop: %s' % item)
            if item.dead:
                continue
            if logger.level >= 4:
                logger.writeln('new items:')
            for new_item in self.unary_deduce(item):
                self.unary_edges_proposed += 1
                if logger.level >= 4:
                    logger.writeln(new_item)
                if self.chart.add(new_item):
                    if logger.level >= 4:
                        logger.writeln('added')
                    heappush(queue, new_item)
                else:
                    if logger.level >= 4:
                        logger.writeln('not added')

    def unary_deduce(self, item):
        # TODO: this is made compatible with both LexicalITG and SCFG for now
        grammars = []
        for grammar in self.grammars:
            if type(grammar) is LexicalITG:
                grammars.append(grammar.itg)
            else:
                grammars.append(grammar)
        for grammar in grammars:
            nt = item.var
            if FLAGS.nt_mismatch:
                nt = nocat(item.var)
            for rule in grammar.iter_rules((nt,)):
                # derive start symbol only for spans [0,i]
                if rule.lhs == FLAGS.start_symbol and item.i != 0:
                    continue
                # derive goal symbol only for span [0,N]
                if (rule.lhs == FLAGS.goal_symbol and
                    not (item.i == 0 and item.j == self.N)):
                    continue
                new_item =  self.features.make_new_item(rule,
                                                        (item,),
                                                        item.i,
                                                        item.j)
                yield new_item

    def get_cube_op(self, i, j):
        """Return a cube operator (a function) that takes as input a list
        [rule, item1, item2, ...]
        and returns a new item. The reason to generate a function on the fly
        here is to hide span for new item (i,j) in the generated function, so
        that it conforms to the cube operator interface."""
        def cube_op(operands):
            rule = operands[0]
            items = operands[1:]
            return self.features.make_new_item(rule, items, i, j)
        return cube_op

    def get_goal(self):
        # TODO: ugly
        # TODO: there can be more than one goal item
        if 'GOAL' in self.chart.bins:
            goal_items = [item for item in self.chart.bins['GOAL']]
            goal_items.sort()
            if len(goal_items) > 0:
                return goal_items[0]
        #else:
        #    # TODO: this is probably wrong
        #    goal_items = [item for item in self.chart.items(0, self.N)]
        #    goal_items.sort()
        #    if len(goal_items) > 0:
        #        return goal_items[0]

    def iter_unknown_items(self):
        # unknown word rules are genereted for all words, because if rules
        # produce only intermediate nonterminals for the word, the decoder may 
        # still get stuck
        for i in range(self.N):
            rule = self.unknown_word_rule(self.fwords[i])
            item = self.features.make_new_item(rule, (), i, i+1)
            yield item
