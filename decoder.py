#!/usr/bin/env python

import sys
from time import sleep
import math
import threading
import os
import socket

import gflags
FLAGS = gflags.FLAGS
from hypergraph import Edge, Node, Hypergraph
from grammar import LexicalITG, SCFG
from corpus_server import CorpusServer
from monitor import Monitor
from consensus_training import ConsensusTrainer, NgramCounter
from references import References
from translation_job import TranslationJob
from features import Features
import logger
from job_logger import JobLogger

gflags.DEFINE_boolean(
    'consensus_training',
    False,
    'Switch for consensus training.')
gflags.DEFINE_integer(
    'cpu',
    25,
    'Percentage of CPU reserved for a slave.')
gflags.DEFINE_enum(
    'decoding_method',
    'agenda',
    ['agenda', 'cyk', 'earley'],
    'Method used for decoding.')
gflags.DEFINE_string(
    'run_dir',
    'run-dir',
    'Directory for dumping decoding results.')
gflags.DEFINE_list(
    'features',
    None,
    'Features used in decoding. (e.g "LM:1,LengthPenalty:-0.2")')
gflags.MarkFlagAsRequired('features')
gflags.DEFINE_list(
    'grammars',
    None,
    'List of grammar files used in decoding.')
gflags.DEFINE_enum(
    'heuristic',
    'cyk',
    ['cyk', 'best_first'],
    'Heuristic function used for agenda-based decoding.')
gflags.DEFINE_string(
    'input',
    None,
    'Input file for decoding. One sentence per line.')
gflags.MarkFlagAsRequired('input')
gflags.DEFINE_integer(
    'kbest_k',
    100,
    'Number of K-best output.')
gflags.DEFINE_string(
    'kbest_output',
    'kbest',
    'Kbest output file.')
gflags.DEFINE_string(
    'lm',
     None,
    'Language model file.')
def lm_checker(value):
    if 'LM' in FLAGS.features:
        return value is not None
    else:
        return True
gflags.RegisterValidator(
     'lm',
     lm_checker,
     message='Specify LM file if you use LM feature.',
     flag_values=FLAGS)
gflags.DEFINE_integer(
    'lm_order',
    3,
    'Language model order.')
gflags.DEFINE_string(
    'log',
    'stderr',
    'Log output.')
gflags.DEFINE_integer(
    'mem',
    1000000,
    'Amount of memory reserved for a slave in KB.')
gflags.DEFINE_list(
    'nodes',
    'f02,f03,f04',
    'List of slave hostnames for parallel decoding.')
gflags.DEFINE_string(
    'output',
    'output',
    'Decoding output file.')
gflags.DEFINE_boolean(
    'output_kbest',
    False,
    'Ouput Kbest translations.')
gflags.DEFINE_boolean(
    'parallel',
    False,
    'Switch for parallel decoding.')
gflags.DEFINE_string(
    'refs',
    None,
    'Prefix of reference files.')
gflags.DEFINE_integer(
    'rule_bin_size',
    5,
    'Number of rules kept for rules with the same French side.')
gflags.DEFINE_string(
    'server',
    None,
    'Server hostname (used by slaves during parallel decoding).')
gflags.DEFINE_boolean(
    'show_time',
    False,
    'Display decoding time.')
gflags.DEFINE_boolean(
    'slave',
    False,
    'Run this decoder instance as slave. For parallel decoding.')
gflags.DEFINE_string(
    'slave_id',
    None,
    'Slave id (used by slaves during parallel decoding).')
gflags.DEFINE_integer(
    'v',
    1,
    'Verbosity level.')
gflags.DEFINE_string(
    'weight_range',
    None,
    'Ranges for feature weights.')

#TODO:
# - insertion rules
# - log ratio pruning
# - LM heuristic 

def get_jobs(range):
    """range is a string specifying sentences to be decoded""" 
    if range:
        if '-' in range:
            start, end = range.split('-')
            start = int(start)
            end = int(end)
            do = lambda x: start <= x <= end
        else:
            number = int(range)
            do = lambda x: x == number
    else:
        do = lambda x: True

    f = open(FLAGS.input)
    jobs = []
    for i, line in enumerate(f):
        if do(i+1):
            job = TranslationJob(i+1,
                                 line,
                                 grammars,
                                 decoding_features)
            jobs.append(job)
    f.close()
    return jobs

def single_worker_decode():
    jobs = get_jobs(FLAGS.do)
    njobs = len(jobs)
    fout = open(FLAGS.output, 'w')
    if FLAGS.output_kbest:
        fkbest = open(FLAGS.kbest_output, 'w')
    totaltime = 0
    joblogger = JobLogger()

    while jobs:
        # finished jobs need to be discarded because jobs save the hypergraph
        job = jobs.pop(0)
        job.run()
        totaltime += job.time
        joblogger.log(job.id)

    if logger.level >= 1 and FLAGS.show_time:
        logger.writeln('total time: %.2f seconds (%.2f seconds/sentence)' %
                       (totaltime, totaltime/njobs))
    joblogger.finish()
    if FLAGS.consensus_training:
        consensus_trainer = ConsensusTrainer(FLAGS.lm_order,
                                             decoding_features,
                                             FLAGS.run_dir,
                                             refs)
        consensus_trainer.optimize()

def tell_server(data, recv=False):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((FLAGS.server, FLAGS.port))
    sock.send((data + '\n').encode())
    if recv:
        received = sock.recv(1024)
    sock.close()
    if recv:
        return received.decode()

def slave_mode_decode():
    # give master time to set up other slaves and start serving
    sleep(5) 
    while True:
        range = tell_server('%s %s' % (FLAGS.slave_id, 'ready'),
                            recv=True).strip()
        if range == '0':  # no more sentences to decode
            return
        jobs = get_jobs(range)
        while jobs:
            job = jobs.pop(0)
            job.run()
        tell_server('%s ok' % FLAGS.slave_id)

grammars = None
decoding_features = None

def main():
    global grammars
    global decoding_features

    logger.level = FLAGS.v
    if FLAGS.log == 'stdout':
        logger.file = sys.stdout
    elif FLAGS.log == 'stderr':
        logger.file = sys.stderr
    else:
        logger.file = open(FLAGS.log, 'w')

    if not FLAGS.slave:
        # clean partial result dir
        os.system('rm -rf %s' % FLAGS.run_dir)
        os.system('mkdir %s' % FLAGS.run_dir)

        # print and save a record of flags
        config_record = open('%s/%s' % (FLAGS.run_dir, 'run.flag'), 'w')
        flags_str = FLAGS.FlagsIntoString()
        config_record.write(flags_str)
        config_record.close()
        if logger.level >= 1:
            logger.write(flags_str)

    if FLAGS.refs is not None:
        refs = References(FLAGS.refs, FLAGS.lm_order)

    if not FLAGS.parallel or FLAGS.slave:
        decoding_features = Features(FLAGS.features, FLAGS.weight_range)

        grammars = []
        for grammar_file in FLAGS.grammars:
            if FLAGS.decoding_method == 'earley':
                grammar = SCFG(grammar_file,
                               FLAGS.rule_bin_size,
                               decoding_features)
            else:
                grammar = LexicalITG(grammar_file,
                                     FLAGS.rule_bin_size,
                                     decoding_features)
            grammars.append(grammar)
        for grammar in grammars:
            grammar.initialize()

    if FLAGS.parallel:  # as master
        # probe nodes
        available_nodes = []
        for node in FLAGS.nodes:
            m = Monitor(node)
            if logger.level >= 1:
                logger.writeln('node %s' % node)
            n_instances = m.probe(FLAGS.cpu, FLAGS.mem)
            available_nodes += ([node] * n_instances)

        if logger.level >= 1:
            logger.writeln('decoding with %s slaves' % len(available_nodes))

        server = CorpusServer(available_nodes)
        server.run()
        if FLAGS.consensus_training:
            decoding_features = Features(FLAGS.features)
            consensus_trainer = ConsensusTrainer(FLAGS.lm_order,
                                                 decoding_features,
                                                 FLAGS.run_dir,
                                                 refs)
            consensus_trainer.optimize()
    else:
        if FLAGS.slave:  # as slave
            slave_mode_decode()
        else:  # as lone decoder
            single_worker_decode()

if __name__ == '__main__':
    try:
        argv = FLAGS(sys.argv)  # parse flags
    except gflags.FlagsError as e:
        print('%s\nUsage: %s ARGS\n%s' % (e, sys.argv[0], FLAGS))
        sys.exit(1)
    main()
