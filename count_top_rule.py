#!/usr/bin/env python3

import sys

from rule import Rule
from percent_counter import PercentCounter
from em_trainer import is_glue

if __name__ == '__main__':
    filenames = sys.argv[1:]
    pcounter = PercentCounter(total=len(filenames), file=sys.stderr)
    nsuper = 0
    for i, fname in enumerate(filenames):
        pcounter.print_percent(i)
        f = open(fname)
        tmp = next(f)
        line = next(f)
        f.close()
        rule = Rule()
        rule.fromstr(line)
        if rule.arity > 1:
            nsuper += 1
    print('%s/%s super/all' % (nsuper, len(filenames)))
