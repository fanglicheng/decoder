#!/usr/bin/env python

# alignment.py
# David Chiang <chiang@isi.edu>

# Copyright (c) 2004-2006 University of Maryland. All rights
# reserved. Do not redistribute without permission from the
# author. Not for commercial use.

import sys
import re

def get_index_map(cover):
    """Return an index map given a coverage vector.

    >>> get_index_map([False, True, True, False, True])
    {1: 0, 2: 1, 3: 2, 4: 2, 5: 3}
    """
    result = {}
    j = 0
    for i, bools in enumerate(zip([False] + cover, cover + [False])):
        if bools[0] or bools[1]:
            result[i] = j
        if bools[1]:
            j += 1
    return result

def get_reversed_index_map(cover):
    """Return a reversed index map given a coverage vector.

    >>> get_reversed_index_map([False, True, True, False, True])
    [[0, 1], [2], [3, 4], [5]]
    """
    result = [[]]
    for i, bools in enumerate(zip([False] + cover, cover + [False])):
        result[-1].append(i)
        if bools[1]:
            result.append([])
    return result

class Alignment(object):
    def __init__(self, fwords, ewords, comment=None):
        m = len(fwords)
        n = len(ewords)
        self.aligned = [[0 for j in range(n)] for i in range(m)]
        self.faligned = [0]*m
        self.ealigned = [0]*n
        self.fwords = fwords
        self.ewords = ewords
        self.comment = comment
        self.espans = None
        
    eline_re = re.compile(r"([^\s]+)\s+\(\{\s+((?:\d+\s+)*)\}\)")

    def remove_unaligned(self):
        fwords = [w for w, a in zip(self.fwords, self.faligned) if a]
        ewords = [w for w, a in zip(self.ewords, self.ealigned) if a]
        new_align = Alignment(fwords, ewords)
        new_align.aligned = [[x for x, ea in zip(row, self.ealigned) if ea]
                             for row, fa in zip(self.aligned, self.faligned) if fa]
        new_align.faligned = [1]*len(fwords)
        new_align.ealigned = [1]*len(ewords)
        return new_align

    def remove_uncovered_words(self, phrases):
        """Remove words not covered by any phrase."""
        # compute cover
        fcover = [False] * len(self.fwords)
        ecover = [False] * len(self.ewords)
        for phrase in phrases:
            fi, ei, fj, ej = phrase
            for k in range(fi, fj):
                fcover[k] = True
            for k in range(ei, ej):
                ecover[k] = True
        # change self 
        self.fwords = [w for w, cover in zip(self.fwords, fcover) if cover]
        self.ewords = [w for w, cover in zip(self.ewords, ecover) if cover] 
        self.faligned = [w for w, cover in zip(self.faligned, fcover) if cover] 
        self.ealigned = [w for w, cover in zip(self.ealigned, ecover) if cover]
        self.aligned = [[x for x, ec in zip(row, ecover) if ec]
                        for row, fc in zip(self.aligned, fcover) if fc]
        # generate new phrases
        old_fi2new_fi = get_index_map(fcover)
        old_ei2new_ei = get_index_map(ecover)
        new_phrases = []
        for phrase in phrases:
            fi, ei, fj, ej = phrase
            new_phrases.append( (old_fi2new_fi[fi],
                                 old_ei2new_ei[ei],
                                 old_fi2new_fi[fj],
                                 old_ei2new_ei[ej]) )
        return new_phrases

    def remove_unaligned_words(self, phrase_nodes):
        "This consolidates nodes on a hypergraph."
        old_fi2new_fi = get_index_map(self.faligned)
        old_ei2new_ei = get_index_map(self.ealigned)
        for node in phrase_nodes:
            node.fi = old_fi2new_fi[node.fi]
            node.fj = old_fi2new_fi[node.fj]
            node.ei = old_ei2new_ei[node.ei]
            node.ej = old_ei2new_ei[node.ej]
        m = sum(self.faligned)
        n = sum(self.ealigned)
        return m, n

    def reader(file, transpose=False):
        while True:
            try:
                comment = file.next().rstrip()
                ewords = file.next().split()
                fline = file.next()
                fxwords = Alignment.eline_re.findall(fline)
            except StopIteration:
                return
            (fword, eindices) = fxwords[0]
            if fword == "NULL":
                fxwords = fxwords[1:]
            fxwords = [(fword, eindices) for (fword, eindices) in fxwords]
            fwords = [fword for (fword, eindices) in fxwords]

            if not transpose:
                a = Alignment(fwords, ewords, comment)
            else:
                a = Alignment(ewords, fwords, comment)

            for i in range(len(fxwords)):
                (fword, eindices) = fxwords[i]
                for eindex in eindices.split():
                    j = int(eindex)-1
                    if not transpose:
                        a.align(i,j)
                    else:
                        a.align(j,i)

            yield a
    reader = staticmethod(reader)

    aline_re = re.compile(r"(\d+)-(\d+)")

    def reader_pharaoh(ffile, efile, afile):
        for (fline, eline, aline) in zip(ffile, efile, afile):
            fwords = fline.split()
            ewords = eline.split()
            a = Alignment(fwords, ewords)
            for (i,j) in Alignment.aline_re.findall(aline):
                i = int(i)
                j = int(j)
                if i >= len(fwords) or j >= len(ewords):
                    sys.stderr.write("warning: alignment point (%s,%s) out of bounds (%s,%s)\n" % (i,j,len(fwords),len(ewords)))
                    continue
                a.align(i,j)
            yield a
    reader_pharaoh = staticmethod(reader_pharaoh)

    def write(self, file):
        '''Write in GIZA++ format'''
        file.write("%s\n" % self.comment)
        file.write("%s\n" % " ".join([sym.tostring(word) for word in self.ewords]))
        output = []
        output += ['NULL','({']+[str(j+1) for j in range(len(self.ewords)) if not self.ealigned[j]]+['})']
        for i in range(len(self.fwords)):
            output += [sym.tostring(self.fwords[i]),'({']+[str(j+1) for j in range(len(self.aligned[i])) if self.aligned[i][j]]+['})']
        file.write("%s\n" % " ".join(output))

    def write_pharaoh(self, file):
        '''Write in Pharaoh format'''
        output = []
        for i in range(len(self.fwords)):
            for j in range(len(self.ewords)):
                if self.aligned[i][j]:
                    output += ['%d-%d' % (i,j)]
        file.write("%s\n" % " ".join(output))

    def write_visual(self, file):
        file.write(" ")
        for j in range(len(self.ewords)):
            file.write("%d" % (j % 10))
        file.write("\n")
        for i in range(len(self.fwords)):
            file.write("%d" % (i % 10))
            for j in range(len(self.ewords)):
                if self.aligned[i][j]:
                    file.write("*")
                else:
                    file.write(".")
            file.write("\n")

    def align(self, i, j):
        if not self.aligned[i][j]:
            self.aligned[i][j] = 1
            self.faligned[i] = 1
            self.ealigned[j] = 1

    def intersect(a1, a2):
        a = Alignment(a1.fwords, a1.ewords)
        for i in range(len(a.fwords)):
            for j in range(len(a.ewords)):
                if a1.aligned[i][j] and a2.aligned[i][j]:
                    a.align(i,j)
        return a
    intersect = staticmethod(intersect)

    def union(a1, a2):
        a = Alignment(a1.fwords, a1.ewords)
        for i in range(len(a.fwords)):
            for j in range(len(a.ewords)):
                if a1.aligned[i][j] or a2.aligned[i][j]:
                    a.align(i,j)
        return a
    union = staticmethod(union)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
