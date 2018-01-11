#!/usr/bin/env python3

import gflags
FLAGS = gflags.FLAGS

gflags.DEFINE_integer(
    'maxabslen',
    10,
    'Maximum initial base phrase length.')
gflags.DEFINE_integer(
    'maxlen',
    5,
    'Maximum final phrase length.')
gflags.DEFINE_integer(
    'minhole',
    1,
    'Minimum sub phrase length.')
gflags.DEFINE_integer(
    'maxvars',
    2,
    'Maximum number of nonterminals.')
gflags.DEFINE_boolean(
    'tight',
    True,
    'Use tight phrases only.')
gflags.DEFINE_list(
    'weightfiles',
    None,
    'Lexical weight tables')
gflags.DEFINE_list(
    'parallel_corpus',
    None,
    'Foreign, English, Alignment.')
gflags.DEFINE_boolean(
    'remove_overlaps',
    False,
    'Remove overlapping phrases.')
gflags.DEFINE_boolean(
    'forbid_adjacent',
    True,
    'Forbid adjcacent nonterminals.')
gflags.DEFINE_boolean(
    'pharaoh',
    True,
    'Pharaoh-style input.')
gflags.DEFINE_boolean(
    'etree_labels',
    False,
    'Relabel using English tree.')
gflags.DEFINE_boolean(
    'require_aligned_terminal',
    True,
    'Require aligned terminal.')
gflags.DEFINE_boolean(
    'keep_word_alignments',
    False,
    'Keep word alignments.')
gflags.DEFINE_string(
    'hypergraph',
    None,
    'Hypergraph directory.')
