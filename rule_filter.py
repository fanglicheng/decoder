#!/usr/bin/env python3

import gflags
FLAGS = gflags.FLAGS

from rule import isvar
import rule_extraction_flags

gflags.DEFINE_integer(
    'compose_level',
    5,
    'Maximal number of minimal rules used to make composed rules.')

def filter(rule):
    #if rule.level > FLAGS.compose_level:
    #    return False
    if rule.arity > 2:
        return False
    if len(rule.f) > 5:
        return False
    # filter adjacent nonterminals
    nts = [isvar(sym) for sym in rule.f]
    for a, b in zip([False] + nts, nts + [False]):
        if a and b:
            return False
    return True

def filter_box(parent, children, align):
    if len(children) > 2:
        return False
    for i in range(len(children)-1):
        if children[i].fj == children[i+1].fi:
            return False
    l = parent.fj - parent.fi
    for c in children:
        l -= c.fj - c.fi
    if l + len(children) > 5:
        return False
    if FLAGS.require_aligned_terminal:
        # TODO: implementation not consistent with hiero
        c = 0
        inchild = False
        for i in range(parent.fi, parent.fj):
            if c < len(children) and i == children[c].fj:
                inchild = False
                c += 1
            if c < len(children) and i == children[c].fi:
                inchild = True
            if not inchild and align.faligned[i]:
                break
        else:
            return False
    return True
