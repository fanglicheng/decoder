#!/usr/bin/env python3

import os
import sys

import logger
import gflags
from rule import Rule
FLAGS = gflags.FLAGS

gflags.DEFINE_integer(
    'dump_size',
    1000000,
    'Rule dumper dumps rules after seeing #dump_size# rule types.')
gflags.DEFINE_string(
    'rule_dump',
    'rule-dump',
    'Directory to dump rules.')
gflags.DEFINE_bool(
    'accumulate',
    True,
    'Accumulate rule occurrences before dump.')

class RuleDumper(object):
    def __init__(self,
                 outputdir=None,
                 dump_size=None,
                 parallel=False):
        # __init__ parameters override FLAGS
        self.outputdir = FLAGS.rule_dump
        self.dump_size = FLAGS.dump_size
        if outputdir is not None:
            self.outputdir = outputdir
        if dump_size is not None:
            self.dump_size = dump_size

        self.parallel = parallel

        os.system('rm -rf %s' % self.outputdir)
        os.mkdir(self.outputdir)
        self.dumped = 0
        self.n_dump = 1
        if FLAGS.accumulate:
            self.gram = {}
        else:
            self.gram = []

    def add(self, rules):
        """count rules"""
        for r in rules:
            self.add_rule(r)

    def add_rule(self, r):
        if FLAGS.accumulate:
            existing = self.gram.get(r, None)
            if existing is not None:
                feats = [f1+f2 for f1,f2 in zip(existing.feats, r.feats)]
                existing.feats = feats
            else:
                self.gram[r] = r
        else:
            self.gram.append(r)
        if len(self.gram) >= self.dump_size:
            self.dump()

    def iter_rules(self):
        if FLAGS.accumulate:
            for r in self.gram.keys():
                yield r
        else:
            for r in self.gram:
                yield r

    def dump(self):
        """Rules are sorted by the English side. Remember to call this before
        finishing."""
        if self.parallel:
            name = "%04d.%04d" % (self.parallel[0], self.n_dump)
        else:
            name = "%04d" % self.n_dump
        if logger.level >= 1:
            logger.writeln('dumping %s...' % name)
        self.dumped += len(self.gram)
        lines = []
        for r in self.iter_rules():
            lines.append("%s ||| %s\n" % (' '.join(str(s) for s in r.e),
                                          str(r)))
        lines.sort()
        outfile = open(os.path.join(self.outputdir, "extract.%s" % name), "w")
        for line in lines:
            outfile.write(line)
        outfile.close()
        if logger.level >= 1:
            logger.writeln('dumped: %s' % self.dumped)

        if FLAGS.accumulate:
            self.gram = {}
        else:
            self.gram = []
        self.n_dump += 1

if __name__ == '__main__':
    try:
        argv = FLAGS(sys.argv)  # parse flags
    except gflags.FlagsError as e:
        print('%s\nUsage: %s ARGS\n%s' % (e, sys.argv[0], FLAGS))
        sys.exit(1)

    rule_dumper = RuleDumper()

    for i, line in enumerate(sys.stdin, 1):
        rule = Rule()
        rule.fromstr(line)
        rule_dumper.add_rule(rule)

    rule_dumper.dump()
