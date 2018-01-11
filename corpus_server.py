#!/usr/bin/env python3

import socket
import os
from os import system
import sys
from time import time

import gflags
FLAGS = gflags.FLAGS
import logger
from job_logger import JobLogger

gflags.DEFINE_integer(
    'port',
    8940,
    'Port number used in parallel decoding.')

class CorpusServer(object):
    def __init__(self, nodes):
        self.nodes = nodes
        self.host = ''
        self.i = 1  # next job to be assigned
        self.nfinished = 0  # number of finished jobs
        self.nterminated = 0  # number of slaves terminated
        self.sid2job = {}  # a map from slave ids to (job, time)

        # read input file
        f = open(FLAGS.input)
        self.input = f.readlines()
        f.close()
        self.input = [l.strip() for l in self.input]
        self.njobs = len(self.input)  # number of jobs

        self.joblogger = JobLogger()

        self.start_time = time()
        self.total_time = 0

    def run(self):
        self.start_slaves()
        self.serve()
        self.finish()

    # ----------- begin of methods class users usually do not need------------

    def start_slaves(self):
        for sid, node in enumerate(self.nodes):
            self.start_slave(sid, node)

    def start_slave(self, sid, host):
        if logger.level >= 1:
            logger.writeln('start slave %s on %s' % (sid, host))
        cmd = ' '.join(sys.argv)
        # slaves inherit master options but it's important to override _Parallel
        # and _Slave to make them work in slave mode
        # they write their detailed translation report to the same log file
        # but their stdout and stderr are still conviniently connected to the
        # master terminal
        options = "--noparallel --slave --slave_id=%s \
                   --log=slaves.log \
                   --server=%s" % (sid, socket.gethostname())
        system(r'ssh %s "cd %s; nohup %s %s" &' % 
               (host, os.getcwd(), cmd, options))

    def serve(self):
        if logger.level >= 1:
            logger.writeln('start server')
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.host, FLAGS.port))
        self.sock.listen(5)
        while self.nfinished != self.njobs or \
              self.nterminated != len(self.nodes):
            conn, addr = self.sock.accept()
            self.handle(conn)

    def handle(self, conn):
        msg = conn.recv(1024).decode()
        sid, status = msg.split()
        if status == 'ready':
            if self.i <= self.njobs:
                conn.send(('%s\n' % self.i).encode())
                self.sid2job[sid] = (self.i, time())
                self.i += 1
            else:
                conn.send('0'.encode())
                self.nterminated += 1
                if logger.level >= 1:
                    logger.writeln()
                    logger.writeln('slave %s told to terminate' % sid)
        elif status == 'ok':
            self.nfinished += 1
            jid, start_time = self.sid2job[sid]
            self.total_time += time() - start_time
            self.joblogger.log(jid)

    def finish(self):
        if logger.level >= 1:
            logger.writeln()
            logger.writeln('total time: %.2f seconds (%.2f seconds/sentence)' %
                           (self.total_time, self.total_time/self.njobs))
            logger.writeln('real time: %.2f seconds' %
                           (time() - self.start_time))
        self.joblogger.finish()

