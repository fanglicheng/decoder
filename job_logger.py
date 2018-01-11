#!/usr/bin/env python3
import os
from os import system

import gflags
FLAGS = gflags.FLAGS
import logger

class JobLogger(object):
    def log(self, jid):
        """read and write result of one job"""
        if logger.level >= 1:
            fname = '%s/%s_%s' % (FLAGS.run_dir, 'log', str(jid).rjust(5, '0'))
            f = open(fname)
            for line in f:
                logger.write(line)
            logger.writeln()
            f.close()

    def finish(self):
        """collect translation/kbest results"""
        files = os.listdir(FLAGS.run_dir)
        outputfiles = ['%s/%s' % (FLAGS.run_dir, f)
                       for f in files if f.startswith(FLAGS.output)]
        kbestfiles = ['%s/%s' % (FLAGS.run_dir, f)
                      for f in files if f.startswith(FLAGS.kbest_output)]
        outputfiles.sort()
        kbestfiles.sort()
        system('cat %s > %s' % (' '.join(outputfiles), FLAGS.output))
        if FLAGS.output_kbest:
            system('cat %s > %s' % (' '.join(kbestfiles), FLAGS.kbest_output))
        # if logger.level >= 1 and self.showtime:
        #     logger.writeln()
        #     logger.writeln('total time: %.2f seconds (%.2f seconds/sentence)' %
        #                    (self.total_time, self.total_time/self.njobs))
        #     logger.writeln('real time: %.2f seconds' %
        #                    (time() - self.start_time))
