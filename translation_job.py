from time import time
import socket
import re

import gflags
FLAGS = gflags.FLAGS
from decode import Decoder
import hypergraph
from hypergraph import Hypergraph
import logger
from consensus_training import NgramCounter

gflags.DEFINE_boolean(
    'output_hypergraph',
    False,
    'Output hypergraph files.')
gflags.DEFINE_boolean(
    'reverse_kbest_feature_score',
    True,
    'Output negative feature costs in Kbest output. Turn this to true when \
    the weight tuner assumes the decoder maximizes (as opposed to minimize) \
    model scores.')
gflags.DEFINE_boolean(
    'preprocess',
    False,
    'Test set is marked with $number and $date.')

pattern = re.compile(r'(\$number|\$date)\s*\{\s*(.*?)\s*\}')

def preprocess(line):
    """Input: a sentence with $number and $dates tokens.
       Output: the sentence with brackets removed and a map for special
               symbols"""
    return pattern.sub(r'\1', line), pattern.findall(line)

class TranslationJob(object):
    def __init__(self,
                 id,
                 line,
                 grammars,
                 features):
        self.id = id
        if FLAGS.preprocess:
            self.line, self.special = preprocess(line)
        else:
            self.line = line
        self.grammars = grammars
        self.features = features
        self.out = None   # output
        self.kbest = None
        self.time = None
        self.ref = None  # reference, class RefCounter

        self.suffix = str(id).rjust(5, '0')

    def run(self):
        # update per-sentence grammars, if there's any
        for g in self.grammars:
            g.update(self.id)
        self.flog = open('%s/%s_%s' % (FLAGS.run_dir,
                                  'log',
                                  self.suffix),
                    'w')
        if FLAGS.show_time:
            self.flog.write('running on %s\n\n' % socket.gethostname())
            self.flog.flush()

        fwords = self.line.split()
        if FLAGS.preprocess:
            self.fidx2replacement = {}
            j = 0
            for i, token in enumerate(fwords):
                if token in ('$number', '$date'):
                    self.fidx2replacement[i] = self.special[j][1]
                    j += 1

        self.flog.write('[%s][%s words] %s\n' %
                   (self.id, len(fwords), self.line))
    
        decoder = Decoder(fwords,
                          self.grammars,
                          self.features)
    
        begin_time = time()
        if FLAGS.decoding_method == 'agenda':
            item = decoder.decode()
        elif FLAGS.decoding_method == 'cyk':
            item = decoder.decode_cyk()
        elif FLAGS.decoding_method == 'earley':
            item = decoder.decode_earley()
        else:
            assert False, '"%s" not valid decoding option' \
                    % FLAGS.decoding_method
        self.time = time() - begin_time

        if item is None:
            self.out = '[decoder failed to build a goal item]'
        else:
            hg = Hypergraph(item)
            hg.set_semiring(hypergraph.SHORTEST_PATH)
            hg.set_functions(lambda x: x.cost, None, None)
            hg.topo_sort()
            self.kbest = hg.root.best_paths()
            output_tokens = self.kbest[0].translation[:]

            if FLAGS.preprocess:
                for i in range(len(output_tokens)):
                    if output_tokens[i] in ('$number', '$date'):
                        fidx = self.kbest[0].composed_rule.we2f[i]
                        if fidx is not None:
                            output_tokens[i] = self.fidx2replacement[fidx]

            self.out = ' '.join(output_tokens[FLAGS.lm_order-1:
                                              1-FLAGS.lm_order])
            self.hg = hg
            if FLAGS.output_hypergraph:
                self.write_hypergraph()

        self.flog.write('%s\n' % self.out)
        self.flog.write('\n')
        if item is not None:
            self.flog.write(self.kbest[0].tree_str())
            self.flog.write('\n')
            self.flog.write(hg.stats())
            self.flog.write('\n')
        self.flog.write(decoder.agenda_stats())
        self.flog.write('\n')
        self.flog.write(decoder.chart.stats())
        self.flog.write('\n')
        for dotchart in decoder.dotcharts:
            self.flog.write(dotchart.stats())
            self.flog.write('\n')

        if FLAGS.show_time:
            timeline = '{:<35}{:>15.2f}\n'.format('[time]:', self.time)
            self.flog.write(timeline)
        self.write_output_file()
        if FLAGS.output_kbest:
            self.write_kbest_to_file()
        self.flog.close()

    def write_output_file(self): 
        fout = open('%s/%s_%s' % (FLAGS.run_dir,
                                  FLAGS.output,
                                  self.suffix),
                    'w')
        fout.write('%s\n' % self.out)
        fout.close()

    def write_kbest_to_file(self):
        fkbest = open('%s/%s_%s' % (FLAGS.run_dir,
                                    FLAGS.kbest_output,
                                    self.suffix),
                      'w')
        if self.kbest is None:
            # TODO: generating an empty kbest list for a particular
            # weight may not be good for MERT, because MERT doesn't get
            # to know this set of weight doesn't work
            self.flog.write('failed to generate kbest\n')
            self.flog.write('\n')
        else:
            for path in self.kbest.iter_top(FLAGS.kbest_k):
                out = ' '.join(x for x in path.translation[FLAGS.lm_order-1:
                                                           1-FLAGS.lm_order])
                if FLAGS.reverse_kbest_feature_score:
                    fcosts = ' '.join(str(-x) for x in path.fcosts)
                else:
                    fcosts = ' '.join(str(x) for x in path.fcosts)
                # zmert requires 0-based indices for input sentences
                fkbest.write('%s\n' % ' ||| '.join([str(self.id - 1),
                                                    out,
                                                    fcosts]))
        fkbest.close()

    def write_hypergraph(self):
        #ngram_counter = NgramCounter(FLAGS.lm_order)
        #ngram_counter.mark_ngrams(self.hg)
        fname = '%s/%s_%s' % (FLAGS.run_dir,
                              'hg',
                              self.suffix)
        if FLAGS.reverse_kbest_feature_score:
            for edge in self.hg.edges():
                edge.fcosts = [-fcost for fcost in edge.fcosts]
        self.hg.serialize(fname)
        self.hg.show()
        if FLAGS.reverse_kbest_feature_score:
            for edge in self.hg.edges():
                edge.fcosts = [-fcost for fcost in edge.fcosts]
