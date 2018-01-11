#!/usr/bin/env python3

import gflags
FLAGS = gflags.FLAGS

gflags.DEFINE_bool(
    'nt_mismatch',
    False,
    'Allow nonterminal mismatch.')
