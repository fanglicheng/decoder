#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Tag

"""Front end for GHKM rule extractor"""

import shutil
import os
import subprocess
import collections
import re
import math

from sys import stderr
from sys import stdin
from datetime import datetime
from collections import defaultdict

LEX_S_GIVEN_T = "/p/mt-scratch2/chung/stages/data/lex-full/lex.c_given_e"
LEX_T_GIVEN_S = "/p/mt-scratch2/chung/stages/data/lex-full/lex.e_given_c"

SRC = "/p/mt-scratch2/chung/stages/train/training.chi"
PARSED_TRG = "/p/mt-scratch2/chung/stages/split-merge/left-bin-removed-7/training.all.labeled_5.mrg.top"
ALIGNMENT = "/p/mt-scratch2/chung/stages/train/training.a"

#SRC = "src"
#PARSED_TRG = "trg"
#ALIGNMENT = "a"

EXTRACTOR_BIN = "frontier_new.pl"
DUMP_SIZE = 100000
NONLEXICAL_RULE_FILE = "nonlexical.scope3.ghkm.gr"
GLUE_RULE_FILE = "a_glue.grammar"
DUMP_FILE = "extract"
DUMP_DIR = "rules-all"

SCOPE_PRUNE = True
MAX_SCOPE = 3
# prune lexical rules with where both lexical weighting less than ZERO
LEXICAL_WEIGHTING_PRUNE = True
ZERO = math.exp(-99)
# prune non-lexical rules that has smoothed count of one
NONLEXICAL_SMOOTHED_PRUNE = False
# do not extract rules with more than max_terminals 
MAX_TERMINALS = 7

# regular expression of rule conversion
re_rule = re.compile("^(\S+)(\(.+) -> (.+) \|\|\|(.+)$")
re_leaves = re.compile("x\d+:[^\)\s]+|\"E_\S+\"")
#re_word = re.compile("\"C_(\S+)\"")
re_nonterminal = re.compile("x(\d+):(\S+)")
re_alignment = re.compile('"E_(\S+)"\-"C_(\S+)"')

# terminal prefixes
SRC_PRE = '"C_'
TRG_PRE = '"E_'

# prints current time
now = lambda: datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S >")

# decide whether rule is lexical or non-lexical given source side of the rule
is_lexical = lambda src: True if SRC_PRE in src else False

# convert '"C_word"' -> word 
remove_prefix = lambda w, prefix: w[3:-1] if w.startswith(prefix) else w

# convert back these special characters
dict_terminals = {"DOUBLE_QT":'"', "-LRB-":"(", "-RRB-":")", "“":'"', "``":'"', "”":'"'}

# convert back if the terminal exists in dict_terminals
convert_terminal = lambda t: dict_terminals[t] if t in dict_terminals else t

# for reading LEX_S_GIVEN_T, LEX_T_GIVEN_S 
dict_src_given_trg = defaultdict(float)
dict_trg_given_src = defaultdict(float)

# calculates scope given source side of the rule
def scope(src_side):
    b = [True, True]
    for child in src_side.split():
        bool_nonterminal = False if child.startswith(SRC_PRE) else True
        b.insert(-1, bool_nonterminal)

    return sum(map(lambda x: 1 if x[0] and x[1] else 0, zip(b[:-1], b[1:])))

# converts nonterminals to new format
def convert_nonterminals(lhs, src_side, trg_side):
    # spaces are added at the end to match last part of the rule
    trg_side = trg_side + " "
    src_side = src_side + " "

    # match trg-side numbered nonterminals with src-side nonterminals
    for index, nonterminal in re_nonterminal.findall(trg_side):
        int_index = int(index)
        old_format_src = "x%d " % (int_index,)
        old_format_trg = "x%d:%s " % (int_index, nonterminal)

        # since comma is delimiter in new format, it needs to be replaced
        nonterminal = nonterminal.replace(",", "COMMA")
        new_format = "[%s,%d] " % (nonterminal, int_index+1)

        src_side = src_side.replace(old_format_src, new_format)
        trg_side = trg_side.replace(old_format_trg, new_format)

    # same case with lhs
    lhs = lhs.replace(",", "COMMA")
    
    # strip out the last space 
    src_side = src_side.strip()
    trg_side = trg_side.strip()

    return lhs, src_side, trg_side

def get_scores(prefix, rule, scores, entire_scores):
    lexical_weight = 1.0

    # if no alignment check p(word|NULL)
    for word in filter(lambda word: word.startswith(prefix), rule.split()):
        word = remove_prefix(word, prefix)
        word = convert_terminal(word)
        if word not in scores:
            scores[word]["NULL"] = entire_scores[" ".join((word, "NULL"))]

    # now you have all the scores, add it up!
    for word in scores:
        tmp = (1.0/float(len(scores[word]))) * sum(scores[word].values())
        lexical_weight *= tmp

    return lexical_weight


def lexical_weighting(src_side, trg_side, alignment):
    dict_t_s = defaultdict(lambda: defaultdict(float))
    dict_s_t = defaultdict(lambda: defaultdict(float))

    # for all alignments
    for trg, src in re_alignment.findall(alignment):
        trg, src = map(convert_terminal, (trg, src))    
        dict_t_s[trg][src] = dict_trg_given_src[" ".join((trg, src))]
        dict_s_t[src][trg] = dict_src_given_trg[" ".join((src, trg))]

    lex_t_given_s = get_scores(TRG_PRE, trg_side, dict_t_s, dict_trg_given_src)
    lex_s_given_t = get_scores(SRC_PRE, src_side, dict_s_t, dict_src_given_trg)

    # not sure if necessary
    del dict_t_s
    del dict_s_t

    return lex_t_given_s, lex_s_given_t

def convert_terminals(rule, prefix):
    return " ".join(map(convert_terminal, 
        map(lambda word: remove_prefix(word, prefix), rule.split())))

def terminal_limit(src, trg):
    num_src = src.count(SRC_PRE)
    num_trg = trg.count(TRG_PRE)

    # no hope of matching all these terminals
    if num_src > MAX_TERMINALS or num_trg > MAX_TERMINALS:
        return True
    else:
        return False

    #small = min(num_src, num_trg)
    #large = max(num_src, num_trg)

    # cannot possibly a good rule
    #if large > small * 7:
    #    return True

# converts a rule into decoder format
def process_rule(str_rule):
    # match rule from the perl script
    m = re_rule.match(str_rule)

    lhs = m.group(1)    
    trg_side = m.group(2)
    src_side = m.group(3)
    alignment = m.group(4)

    # prune if number of terminals is over MAX_TERMINALS
    if terminal_limit(src_side, trg_side) == True:
        return None, None, None, None

    # if scope pruning is true, ignore rules that exceed max-scope
    if SCOPE_PRUNE == True:
        if scope(src_side) > MAX_SCOPE:
            return None, None, None, None

    # decide whether the rule is lexical or non-lexical
    bool_lexical = is_lexical(src_side)

    # discard internal tree structure in trg-side rule
    trg_side = " ".join(re_leaves.findall(trg_side))

    # converts old nonterminal format to new format
    # src: x0 -> [NP,1], trg: x0:NP -> [NP,1]
    lhs, src_side, trg_side = convert_nonterminals(lhs, src_side, trg_side)

    # calculate lexical weighting
    t_given_s, s_given_t = lexical_weighting(src_side, trg_side, alignment)

    if LEXICAL_WEIGHTING_PRUNE == True:
        if t_given_s < ZERO and s_given_t < ZERO:
            return None, None, None, None

    # remove terminal markers and convert back some special characters
    src_side = convert_terminals(src_side, SRC_PRE)
    trg_side = convert_terminals(trg_side, TRG_PRE)

    tuple_rule = (lhs, src_side, trg_side)
    str_rule = "[%s] ||| %s ||| %s" % tuple_rule
    
    return str_rule, t_given_s, s_given_t, bool_lexical

def read_lexical_weighting_file(file_name, dict_lexical_weighting):
    print(now(), "Reading file:", file_name, file=stderr)
    stderr.flush()

    fin = open(file_name, "r")
    for line in fin:
        list_line = line.split()
        if len(list_line) == 3:
            # this way uses less memory than nested dictionary
            word_pair = " ".join(list_line[:2])
            dict_lexical_weighting[word_pair] = float(list_line[2])

    fin.close()
    print(now(), "Finished reading file", file=stderr)
    stderr.flush()

def create_dump_dir(dir_name):
    # delete dump directory if it exists
    shutil.rmtree(dir_name, ignore_errors=True)

    # create dump directory
    os.mkdir(dir_name)    
    print(now(), "Created directory:", dir_name, file=stderr)
    stderr.flush()

def smoothed_rule(rule):
    return re.sub(r"\[(\S+)\-\d+(\S+)?\]", lambda m: "[%s%s]" % 
        (m.group(1), m.group(2) if m.group(2) != None else ""), rule)

def remove_nonterminal_index(trg):
    return re.sub(r"\[(\S+\-?\d?)(,\d+)\]", lambda m: "[%s]" % m.group(1), trg)

def get_target(rule):
    return rule.split(" ||| ")[2]

def get_lhs(rule):
    return rule.split(" ||| ")[0]

def get_src(rule):
    return rule.split(" ||| ")[1]

def write_to_dump_file(dump_dir, dump_file, lexical_rules, dump_no):
    fout = open("%s/%s.%.4d" % (dump_dir, dump_file, dump_no), "w")

#    for rule in sorted(lexical_rules, 
#        key=lambda x: (get_target(x), get_lhs(x), get_src(x))):

    for rule in sorted(lexical_rules, 
        key=lambda x: "%s ||| %s" % (get_target(x), x)):
                
        v = lexical_rules[rule]
        trg = get_target(rule)
        out = "%s ||| %s ||| %f %s %s" % (trg, rule, v[0], str(v[1]), str(v[2]))

        print(out, file=fout)
    
    fout.close()
    print(now(), "dump file: %d" % dump_no, file=stderr)
    stderr.flush()

def write_nonlexical_rules(nonlexical_rule_file, nonlexical_rules):
    # get smoothed rule count
    rules_smoothed = defaultdict(int)
    for rule, count in nonlexical_rules.items():
        rules_smoothed[smoothed_rule(rule)] += count

    fout = open(nonlexical_rule_file, "w")
    f_total = float(sum(nonlexical_rules.values()))
    for rule, val in nonlexical_rules.items():
        smoothed_count = rules_smoothed[smoothed_rule(rule)] 
        if NONLEXICAL_SMOOTHED_PRUNE == True and smoothed_count == 1:
            continue
 
        prob = float(val) / f_total
        prob_smoothed = float(smoothed_count) / f_total
        print("%s ||| %s %s" % (rule, str(prob), str(prob_smoothed)), file=fout)
         
    fout.close()

def write_nonterminal_glue_rules(glue_rule_file, set_lhs):
    fout = open(glue_rule_file, "w")
    for lhs in set_lhs:
        lhs = lhs[1:-1]
        print("[A] ||| [%s,1] ||| [%s,1] ||| 1e-05" % (lhs, lhs), file=fout)
         
    fout.close()


if __name__ == "__main__":
    # create dump directory
    create_dump_dir(DUMP_DIR)
    
    # read for lexical weighting
    read_lexical_weighting_file(LEX_S_GIVEN_T, dict_src_given_trg)
    read_lexical_weighting_file(LEX_T_GIVEN_S, dict_trg_given_src)

    # run extractor
    str_cmd = "%s -nounary -alignments -depthtwo %s %s %s" % (EXTRACTOR_BIN, 
        PARSED_TRG, ALIGNMENT, SRC)
    print(now(), "Running:", str_cmd, file=stderr)
    stderr.flush()
    p = subprocess.Popen(str_cmd, shell=True, stdout=subprocess.PIPE)

    dump_no = 0
    lexical_rules = defaultdict(lambda: [0, 0.0, 0.0]) #count, lex(t|s) lex(s|t)
    nonlexical_rules = defaultdict(int)
    set_lhs = set()

    while 1:
        rule = p.stdout.readline()
        if not rule:
            break
        rule = rule.decode("utf-8")
        if rule: # ignore encoding failures
            rule, t_s, s_t, bool_lexical = process_rule(rule)
            if rule: # ignore pruned rules
                if bool_lexical == False:
                    nonlexical_rules[rule] += 1
                else:
                    lexical_rules[rule][0] += 1
                    lexical_rules[rule][1] = max(t_s, lexical_rules[rule][1])
                    lexical_rules[rule][2] = max(s_t, lexical_rules[rule][2])

                set_lhs.add(get_lhs(rule))
        if len(lexical_rules) > DUMP_SIZE:
            write_to_dump_file(DUMP_DIR, DUMP_FILE, lexical_rules, dump_no)
            dump_no += 1
            lexical_rules.clear()
    
    # left overs
    write_to_dump_file(DUMP_DIR, DUMP_FILE, lexical_rules, dump_no)
    del lexical_rules                
    print(now(), "Finished extracting", file=stderr)
    stderr.flush()

    print(now(), "Writing non-terminal glue rules", file=stderr)
    stderr.flush()
    write_nonterminal_glue_rules(GLUE_RULE_FILE, set_lhs)
    
    print(now(), "Writing non-lexical rules", file=stderr)
    stderr.flush()
    write_nonlexical_rules(NONLEXICAL_RULE_FILE, nonlexical_rules)
    print(now(), "Done!", file=stderr)
    stderr.flush()

