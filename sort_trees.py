#!/usr/bin/env python3

import sys

from rule import Rule
from percent_counter import PercentCounter
from em_trainer import is_glue

if __name__ == '__main__':
    filenames = sys.argv[1:]
    pcounter = PercentCounter(total=len(filenames), file=sys.stderr)
    rules = {}
    for i, fname in enumerate(filenames):
        pcounter.print_percent(i)
        f = open(fname)
        for line in f:
            if '|||' in line:
                rule = Rule()
                rule.fromstr(line)
                if not is_glue(rule):
                    # each rule on viterbi tree is counted as one
                    rule.feats[0] = 1
                    oldrule = rules.get(rule)
                    if oldrule is None:
                        rules[rule] = rule
                    else:
                        oldrule.feats = [f1 + f2 
                                         for f1, f2 in zip(oldrule.feats,
                                                           rule.feats)]
        f.close()
    lines = []
    for rule in rules:
        sortline = '%s ||| %s\n' % (' '.join(str(s) for s in rule.e),
                                    str(rule))
        lines.append(sortline)
    lines.sort()
    for line in lines:
        sys.stdout.write(line)
