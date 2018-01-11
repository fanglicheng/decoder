#!/usr/bin/env python

# lexweights.py
# David Chiang <chiang@isi.edu>

# Copyright (c) 2004-2006 University of Maryland. All rights
# reserved. Do not redistribute without permission from the
# author. Not for commercial use.

# should a word with multiple alignments evenly distribute 1 point among its partners?

import sys, math, itertools

import alignment

# for calculating likelihood ratios. Manning and Schuetze p. 173
# note natural logs
def ll(k,n,x):
    if k == 0:
        return n*math.log(1-x)
    elif n-k == 0:
        return n*math.log(x)
    else:
        return k*math.log(x) + (n-k)*math.log(1-x)

def llr(n,c1,c2,c12):
    p = float(c2)/n
    p1 = float(c12)/c1
    p2 = float(c2-c12)/(n-c1)
        
    return ll(c12,c1,p) + ll(c2-c12,n-c1,p) - ll(c12,c1,p1) - ll(c2-c12,n-c1,p2)

if __name__ == "__main__":
    try:
        import psyco
        psyco.full()
    except ImportError:
        pass
    
    import optparse
    optparser = optparse.OptionParser()
    optparser.add_option('-w', '--weight', nargs=2, dest='weightfiles', help="lexical weight tables")
    optparser.add_option('-W', '--words', nargs=2, dest='words', help="parallel text files (words)")
    optparser.add_option('-P', '--pharaoh', dest='pharaoh', action='store_true', default=False, help="input is Pharaoh-style alignment (requires -W)")
    optparser.add_option('-r', '--ratio', dest='ratiofile', help="likelihood ratio file")
    (opts,args) = optparser.parse_args()

    if opts.words is not None:
        ffilename, efilename = opts.words
        ffile = open(ffilename)
        efile = open(efilename)

    if len(args) == 0:
        args = ["-"]

    if opts.pharaoh:
        if len(args) != 1:
            sys.stderr.write("Can only read in one file in Pharaoh format\n")
        if opts.words is None:
            sys.stderr.write("-W option required for Pharaoh format\n")
        if args[0] == "-":
            input_file = sys.stdin
        else:
            input_file = open(args[0], "r")
        alignments = alignment.Alignment.reader_pharaoh(ffile, efile, input_file)
    else:
        input_files = []
        for arg in args:
            if arg == "-":
                input_files.append(sys.stdin)
            else:
                input_files.append(open(args[0], "r"))
        alignments = itertools.chain(*[alignment.Alignment.reader(input_file) for input_file in input_files])
        # bug: ignores -W option

    if opts.weightfiles is not None:
        fweightfile = open(opts.weightfiles[0], "w")
        eweightfile = open(opts.weightfiles[1], "w")

    if opts.ratiofile is not None:
        ratiofile = open(opts.ratiofile, "w")
        
    fcount = {}
    ecount = {}
    fecount = {}
    count = 0

    """fweightfile (Pharaoh lex.n2f) is in the format
         f e P(f|e)
       eweightfile (Pharaoh lex.f2n) is in the format
         e f P(e|f)
       where either e or f can be NULL.
    """

    progress = 0
    for a in alignments:
        null = "NULL"
        # Calculate lexical weights
        for i in range(len(a.fwords)):
            for j in range(len(a.ewords)):
                if a.aligned[i][j]:
                    count += 1
                    fcount[a.fwords[i]] = fcount.get(a.fwords[i],0)+1
                    ecount[a.ewords[j]] = ecount.get(a.ewords[j],0)+1
                    fecount[(a.fwords[i],a.ewords[j])] = fecount.get((a.fwords[i],a.ewords[j]),0)+1

        for i in range(len(a.fwords)):
            if not a.faligned[i]:
                count += 1
                fcount[a.fwords[i]] = fcount.get(a.fwords[i],0)+1
                ecount[null] = ecount.get(null,0)+1
                fecount[(a.fwords[i],null)] = fecount.get((a.fwords[i],null),0)+1
        for j in range(len(a.ewords)):
            if not a.ealigned[j]:
                count += 1
                fcount[null] = fcount.get(null,0)+1
                ecount[a.ewords[j]] = ecount.get(a.ewords[j],0)+1
                fecount[(null,a.ewords[j])] = fecount.get((null,a.ewords[j]),0)+1

        progress += 1
        if progress % 10000 == 0:
            sys.stderr.write(".")

    # Dump lexical weights
    for (fword,eword) in fecount.keys():
        if opts.ratiofile:
            # f|e
            c12 = fecount[fword,eword]
            c1 = ecount[eword]
            c2 = fcount[fword]
            p = float(c2)/count
            p1 = float(c12)/c1
            p2 = float(c2-c12)/(count-c1)
            ratiofile.write("%s %s %f\n" % (eword, fword, -2*llr(count,ecount[eword],fcount[fword],fecount[fword,eword])))
        if opts.weightfiles:
            fweightfile.write("%s %s %f\n" % (fword, eword, float(fecount[(fword,eword)])/ecount[eword]))
            eweightfile.write("%s %s %f\n" % (eword, fword, float(fecount[(fword,eword)])/fcount[fword]))

    sys.stderr.write("\n")
